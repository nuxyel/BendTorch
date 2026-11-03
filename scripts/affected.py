#!/usr/bin/env python3
"""What a change needs: the make targets and the mutants that read the files a diff touches.

A make target's recipe names the files it reads, and each of those names its own: a .bend file's relative imports and
the paths in its strings, a Python script's imports of its neighbours and the paths in its strings, any path written in
a shell script. A target's closure is every file reachable that way from its recipe and its prerequisites; a change
needs the target when a changed file is in that closure. A mutant of scripts/mutate.py is judged by its gate, whose
source names the files the gate reads, so it is placed the same way (`mutants` below says when it runs).

A change to the Makefile reaches the targets whose rule, whose variables or whose prerequisites' rules it changes, and a
change to a .bend, .py or .sh file that only touches its comment lines needs nothing. What this cannot place, it does not
skip: the CI workflow and this script call for every target, and so do the closure of `make setup` and `make data`
(every target reads the toolchain, the venv and the data), a line every recipe runs under (an export), and a changed
file that no target is seen to read and that is neither documentation nor a tool run by hand. The mutants keep their
own rule (`mutants`); every mutant runs in the full run.

CI (.github/workflows/ci.yml) runs the selection on pushes and pull requests to devel; pull requests to main, the weekly
run, a run by hand and a pushed commit whose message has [full] run everything, every mutant included.

    python3 scripts/affected.py show [BASE [HEAD]]  # make affected: what the change from BASE (origin/devel) needs, and why
    python3 scripts/affected.py self-check          # make affected-check: the selection's own tests (CI runs them first)
    python3 scripts/affected.py ci                  # in CI: the selection into $GITHUB_OUTPUT and the run's summary
"""
import ast
import fnmatch
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# a change to one of these can change any run (a change to the Makefile reaches the targets whose rules it changes)
EVERYTHING = {".github/workflows/ci.yml", "scripts/affected.py"}
# files nothing runs: a change to them alone needs nothing (scripts/check_rts.py, which reads every tracked file, runs on
# every change anyway)
DOCS = ("*.md", "docs/*", "LICENSE", ".gitignore", ".github/ISSUE_TEMPLATE/*")
# what CI's selection may run: make check's parts, make bits, make check-js; check-rts runs on every change
ALWAYS = {"check-rts"}
EXTRA_CI = ["bits", "check-js"]
# the release gates that need this machine (a GPU, SmolLM2's weights, an 8 GiB sparse file, the packages): listed by
# make affected, never run by CI
LOCAL = ["gpu", "smol-gpu", "corpus", "bits-smol", "smol", "mutate-smol", "reader", "package"]
# files a recipe reads without naming them
READS = {"lint": ["package.json", "bun.lock", "bend-lint.json", "tests/lint_template_value_field.bend"]}
# tools run by hand, which no target runs and nothing reads (checked by grep, 2026-11-03): a change to one alone needs
# nothing; a new file nothing reads calls for every target until it is placed here or read by a target
BY_HAND = {"reference/check_smol_hf.py", "reference/corpus_gen.py", "reference/numerics_probe.py",
           "reference/smol_bf16opt_ref.py", "reference/tiny_probe.py", "scripts/block.sh"}
# the implementation, as opposed to what judges it (the tests, the twins, the laws and the specification): a mutant does
# not run on a change elsewhere in the implementation (see `mutants`)
IMPL = {"torch/main.bend", "torch/fast.bend", "torch/gpu.bend", "torch/load.bend", "torch/tensor.bend", "torch/dtype.bend"}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def tracked():
    return set(git("ls-files").split("\n")) - {""}


# ---- the Makefile -------------------------------------------------------------------------------------------------

class Makefile:
    VAR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|\?=|=)\s*(.*)$")
    RULE = re.compile(r"^([A-Za-z0-9_.%-]+(?:[ \t]+[A-Za-z0-9_.%-]+)*)[ \t]*:(?![=:])(.*)$")

    def __init__(self, text):
        self.vars, self.rules, self.other = {}, {}, []
        target = None
        for line in text.split("\n"):
            if line.startswith("\t"):
                if target:
                    self.rules[target][1].append(line[1:])
                continue
            target = None
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = self.VAR.match(line)
            if m and not self.RULE.match(line.split("=", 1)[0] + ":"):
                self.vars[m.group(1)] = m.group(2)
                continue
            m = self.RULE.match(line)
            if m and not m.group(1).startswith("."):
                target = m.group(1)
                self.rules[target] = (m.group(2).split("#", 1)[0], [])
            elif not (m and m.group(1).startswith(".")):
                self.other.append(line)  # export, include, a conditional: what every recipe runs under

    def expand(self, s, depth=0):
        if depth > 20:
            return s
        def sub(m):
            inner = m.group(1)
            if inner.startswith("addprefix "):
                p, _, rest = inner[len("addprefix "):].partition(",")
                return " ".join(p + w for w in self.expand(rest, depth + 1).split())
            if inner in self.vars:
                return self.expand(self.vars[inner], depth + 1)
            return m.group(0)
        return re.sub(r"\$\(((?:[^()]|\([^()]*\))*)\)", sub, s)

    def rule(self, target):
        if target in self.rules:
            pre, rec = self.rules[target]
            return self.expand(pre).split(), [self.expand(r) for r in rec]
        for pat, (pre, rec) in self.rules.items():
            if "%" in pat:
                a, b = pat.split("%", 1)
                if target.startswith(a) and target.endswith(b) and len(target) > len(a) + len(b):
                    stem = target[len(a):len(target) - len(b)]
                    return ([self.expand(p).replace("%", stem) for p in pre.split()],
                            [self.expand(r).replace("$*", stem) for r in rec])
        return None

    def targets(self):
        return [t for t in self.rules if "%" not in t]

    def key(self, target):
        """The rule that makes a target: its own, or the pattern it matches."""
        if target in self.rules:
            return target
        for pat in self.rules:
            if "%" in pat:
                a, b = pat.split("%", 1)
                if target.startswith(a) and target.endswith(b) and len(target) > len(a) + len(b):
                    return pat
        return None

    def raw(self, target):
        k = self.key(target)
        return None if k is None else (self.rules[k][0].strip(), tuple(self.rules[k][1]))

    def vars_used(self, target):
        """Every variable a target's rule reads, through the variables it reads (inside functions too)."""
        r = self.raw(target)
        todo = re.findall(r"\$[({]([A-Za-z_][A-Za-z0-9_]*)", " ".join([r[0], *r[1]])) if r else []
        seen = set()
        while todo:
            v = todo.pop()
            if v in seen or v not in self.vars:
                continue
            seen.add(v)
            todo += re.findall(r"\$[({]([A-Za-z_][A-Za-z0-9_]*)", self.vars[v])
        return seen


def makefile_affected(old_text, new_text, targets):
    """Which of `targets` a change of the Makefile reaches: those whose rule, a variable it reads, or a prerequisite's
    rule changed; None when a line every recipe runs under changed (an export)."""
    old, new = Makefile(old_text), Makefile(new_text)
    if old.other != new.other:
        return None
    var_changed = {v for v in set(old.vars) | set(new.vars) if old.vars.get(v) != new.vars.get(v)}
    memo = {}
    def hit(t, depth=0):
        if t not in memo:
            memo[t] = False
            direct = (old.raw(t) != new.raw(t) or bool(new.vars_used(t) & var_changed)
                      or bool(old.vars_used(t) & var_changed))
            r = new.rule(t)
            memo[t] = direct or (depth < 20 and any(hit(p, depth + 1) for p in (r[0] if r else [])))
        return memo[t]
    return {t for t in targets if hit(t)}


def code(text, f):
    """A file without its comments: Python's by its tokens (a `#` inside a string stays), Bend's and the shell's by their
    `#` lines (a shebang stays); None for other files."""
    if f.endswith(".py"):
        try:
            return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.NL)]
        except (tokenize.TokenError, SyntaxError):
            return text.split("\n")
    if not f.endswith((".bend", ".sh")):
        return None
    return [l.rstrip() for l in text.split("\n") if l.strip() and not (l.lstrip().startswith("#") and not l.startswith("#!"))]


# ---- what a file reads ----------------------------------------------------------------------------------------------

def python_reads(text):
    """A Python file's imports and the strings in its code; its docstrings and comments name files it only describes
    (the twin of demo/tiny_llama.bend)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return re.findall(r"^\s*(?:from|import)\s+(\w+)", text, re.M), re.findall(r"""["']([^"'\n]+)["']""", text)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    names, lits = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            lits.append(node.value)
    return names, lits


class Graph:
    def __init__(self, files, mk, use_imports=True):
        self.files, self.mk, self.use_imports = files, mk, use_imports
        self.dirs = {}
        for f in files:
            d = os.path.dirname(f)
            while d:
                self.dirs.setdefault(d, []).append(f)
                d = os.path.dirname(d)
        self.demos = {os.path.basename(f)[:-5] for f in files if f.startswith("demo/") and f.endswith(".bend")}
        self.cache = {}

    def paths(self, token, here):
        """The tracked files a token written in a file of directory `here` can name."""
        token = token.strip("'\"`,;()[]{}<>|&")
        if not token:
            return set()
        out = set()
        cands = {token, re.sub(r"^(\.\.?/)+", "", token)}
        if here:
            cands.add(os.path.normpath(os.path.join(here, token)))
        for c in cands:
            c = c.rstrip("/")
            if c in self.files:
                out.add(c)
            elif c in self.dirs and "/" in token.rstrip("/") and not token.endswith("/"):
                # a folder written as a path (bench/corpus), not a bare word (reference) or the start of one built
                # from a variable (demo/$d.bend)
                out.update(self.dirs[c])
            elif re.fullmatch(r"(?:\.\.?/)*[\w-][\w.-]*(?:/[\w.-]+)*/[^/]*[*?][^/]*", token):
                out.update(f for f in self.files if fnmatch.fnmatch(f, c))  # torch/*.bend, not * or ./*
        m = re.search(r"\.scratch/(?:js/)?(?:mut_)?([A-Za-z0-9_]+)", token)
        if m:
            name = m.group(1)
            for _ in range(3):
                if name in self.demos:
                    out.add(f"demo/{name}.bend")
                    break
                name = re.sub(r"_(ref|cuda|gpu|sgd|360m)$", "", name)
        return out

    def scan(self, text, here, words):
        out = set()
        for tok in re.findall(r"[\w./*?-]+", text):
            out |= self.paths(tok, here)
            if words and tok in self.demos:
                out.add(f"demo/{tok}.bend")
        return out

    def edges(self, f):
        if f in self.cache:
            return self.cache[f]
        if f == "scripts/affected.py":
            return set()  # it names files as its data (BY_HAND, IMPL), not as what it reads
        try:
            text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            text = ""
        here, out = os.path.dirname(f), set()
        if f.endswith(".bend"):
            if self.use_imports:
                for p in re.findall(r"^import\s+(\.\.?/\S+)\s+as\b", text, re.M):
                    out |= self.paths(p, here)
            for lit in re.findall(r'"([^"\n]*)"', text):
                out |= self.scan(lit, here, False)
        elif f.endswith(".py"):
            names, lits = python_reads(text)
            for name in names:
                if os.path.join(here, name + ".py") in self.files:
                    out.add(os.path.join(here, name + ".py"))
            for lit in lits:
                out |= self.scan(lit, here, False)
        elif f.endswith(".sh"):
            out |= self.scan(re.sub(r"^\s*#.*$", "", text, flags=re.M), here, True)  # its code, not its comments
        self.cache[f] = out - {f}
        return self.cache[f]

    def closure(self, roots):
        seen, todo = set(), list(roots)
        while todo:
            f = todo.pop()
            if f in seen:
                continue
            seen.add(f)
            todo.extend(self.edges(f) - seen)
        return seen

    def target_roots(self, target, skip=ALWAYS):
        """The files a target's recipe and its prerequisites' recipes name."""
        roots, todo, seen = set(), [target], set()
        while todo:
            t = todo.pop()
            if t in seen or (t in skip and t != target):
                continue
            seen.add(t)
            r = self.mk.rule(t)
            if r is None:
                continue
            pre, rec = r
            todo.extend(pre)
            roots |= self.scan("\n".join(rec), "", True)
            roots |= set(READS.get(t, ())) & self.files
        return roots

    def target_closure(self, target):
        return self.closure(self.target_roots(target))


# ---- the mutants ----------------------------------------------------------------------------------------------------

def load_mutate(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def gate_text(gate):
    src = inspect.getsource(gate)
    for k, v in inspect.getclosurevars(gate).nonlocals.items():
        if isinstance(v, str):
            src = src.replace("{" + k + "}", v)
    return src


def gate_sig(gate):
    nl = inspect.getclosurevars(gate).nonlocals
    return (gate.__name__, inspect.getsource(gate), tuple(sorted((k, v) for k, v in nl.items() if isinstance(v, str))))


def mutate_rest(text):
    """scripts/mutate.py without its list of mutants and its gates: the machinery every mutant runs through."""
    text = re.sub(r"^MUTANTS = \[.*?^\]\n", "", text, flags=re.M | re.S)
    text = re.sub(r"^def (laws|smol_st|fast|gpu|gpu_layers|gpu_expcr|tiny16)\(\):\n(?:(?:[ \t]+.*|)\n)*", "", text, flags=re.M)
    text = re.sub(r"^[A-Z0-9_, ]+ = (?:numeric\(.*\)|[a-z0-9_]+)\n", "", text, flags=re.M)
    return text


# ---- the selection --------------------------------------------------------------------------------------------------

class World:
    def __init__(self, use_imports=True):
        self.files = tracked()
        self.mk = Makefile(open(os.path.join(ROOT, "Makefile")).read())
        self.g = Graph(self.files, self.mk, use_imports)
        check_pre = self.mk.rule("check")[0]
        self.ci_units = []
        for t in check_pre + EXTRA_CI:
            for u in (self.mk.rule(t)[0] if t == "build" else [t]):
                if u not in self.ci_units and u not in ALWAYS:
                    self.ci_units.append(u)
        self.universal = self.g.closure(self.g.target_roots("setup") | self.g.target_roots("data"))
        self.closures = {t: self.g.target_closure(t) | self.universal for t in self.mk.targets() + self.ci_units}
        self.mut = load_mutate(os.path.join(ROOT, "scripts", "mutate.py"), "mutate_head")
        self.gate_files = {}
        for m in self.mut.MUTANTS:
            gate = m[4]
            if gate not in self.gate_files:
                roots = self.g.scan(gate_text(gate), "scripts", False)
                for d in getattr(self.mut, "REFS", {}).get(gate, []):
                    roots.add(f"demo/{d}.bend")
                self.gate_files[gate] = (roots, self.g.closure(roots) | self.universal)

    def place(self, f):
        """('everything' | 'units' | 'docs' | 'by hand' | 'unknown', the CI units, the other targets) for one changed file;
        'everything' and 'unknown' call for every target."""
        if f in EVERYTHING or f in self.universal:
            return "everything", [], []
        ci = [u for u in self.ci_units if f in self.closures[u]]
        other = [t for t in self.mk.targets() if t not in self.ci_units and t not in ALWAYS and f in self.closures[t]]
        if ci or other or f == "scripts/mutate.py" or f in self.closures["check-rts"]:
            return "units", ci, other
        if any(fnmatch.fnmatch(f, p) for p in DOCS):
            return "docs", [], []
        if f in BY_HAND:
            return "by hand", [], []
        return "unknown", [], []

    def mutants(self, changed, base):
        """The mutants a change needs, each with its reason. A mutant measures whether its gate sees a fault in its file,
        so it runs when its file changes, when what judges it changes (its gate's files outside the implementation: the
        test program, the reference script, the twin, the oracle, the laws, the specification), or when it or its gate
        is new or edited in scripts/mutate.py. A change to the machinery of scripts/mutate.py runs them all."""
        old = {}
        if "scripts/mutate.py" in changed:
            try:
                text = git("show", f"{base}:scripts/mutate.py")
            except subprocess.CalledProcessError:
                text = None
            head_text = open(os.path.join(ROOT, "scripts", "mutate.py")).read()
            if text is None or code(mutate_rest(text), "m.py") != code(mutate_rest(head_text), "m.py"):
                return [(m[0], "scripts/mutate.py's machinery changed") for m in self.mut.MUTANTS]
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "mutate.py")
                open(p, "w").write(text)
                base_mod = load_mutate(p, "mutate_base")
                old = {m[0]: (m[1], m[2], m[3], gate_sig(m[4]), m[5]) for m in base_mod.MUTANTS}
        out = []
        for name, path, before, after, gate, expected in self.mut.MUTANTS:
            judges = sorted(f for f in changed & self.gate_files[gate][1] if f not in IMPL)
            if judges:
                out.append((name, f"its gate {gate.__name__} reads {', '.join(judges)}"))
            elif "scripts/mutate.py" in changed and old.get(name) != (path, before, after, gate_sig(gate), expected):
                out.append((name, "new or edited in scripts/mutate.py"))
            elif path in changed:
                out.append((name, f"{path} changed"))
        return out

    def select(self, changed, base, head=None):
        """head: a commit, or None for the files on disk."""
        places = {}
        for f in sorted(changed):
            new, old = content(head, f), content(base, f)
            if f == "Makefile" and new is not None and old is not None:
                hit = makefile_affected(old, new, self.mk.targets() + self.ci_units)
                places[f] = ("everything", [], []) if hit is None else (
                    "units", [u for u in self.ci_units if u in hit],
                    [t for t in self.mk.targets() if t in hit and t not in self.ci_units and t not in ALWAYS])
            elif new is not None and old is not None and code(new, f) is not None and code(new, f) == code(old, f):
                places[f] = ("comments", [], [])
            else:
                places[f] = self.place(f)
        changed = {f for f in changed if places[f][0] != "comments"}
        # a file every run reads, or one nothing is seen to read: every target (the mutants still by their rule; all of
        # them run in the full run)
        broad = any(p[0] in ("everything", "unknown") for p in places.values())
        units = list(self.ci_units) if broad else [u for u in self.ci_units if any(u in p[1] for p in places.values())]
        local = list(LOCAL) if broad else [t for t in LOCAL if any(t in p[2] for p in places.values())]
        return {"broad": broad, "places": places, "units": units, "local": local, "mutants": self.mutants(set(changed), base)}


def content(ref, f):
    if ref is None:
        try:
            return open(os.path.join(ROOT, f), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            return None
    r = subprocess.run(["git", "show", f"{ref}:{f}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        return None
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def changed_files(base, head, worktree=False):
    if not worktree and git("rev-parse", f"{base}^{{tree}}") == git("rev-parse", f"{head}^{{tree}}"):
        return git("rev-parse", base).strip(), set()  # the same files as a commit that passed (a release's merge)
    mb = git("merge-base", base, head).strip()
    out = set()
    for line in git("diff", "--name-status", "--no-renames", mb, head).split("\n"):
        if line.strip():
            out.add(line.split("\t", 1)[1])
    if worktree:
        out |= set(git("diff", "--name-only", "HEAD").split()) | set(git("ls-files", "--others", "--exclude-standard").split())
    return mb, out


def describe(w, sel):
    lines, places = [], sel["places"]
    if sel["broad"]:
        # the files that call for every target; the others add nothing to it
        forcing = {f: p for f, p in places.items() if p[0] in ("everything", "unknown")}
        more, rest = len(forcing) - 12, len(places) - len(forcing)  # the others add nothing to every target
        places = dict(list(forcing.items())[:12])
    for f, (kind, ci, other) in places.items():
        if kind == "everything":
            lines.append(f"  {f}: every run reads it: every target")
        elif kind == "unknown":
            lines.append(f"  {f}: nothing is seen to read it and it is not documentation: every target")
        elif kind == "docs":
            lines.append(f"  {f}: documentation: nothing")
        elif kind == "by hand":
            lines.append(f"  {f}: a tool run by hand: nothing")
        elif kind == "comments":
            lines.append(f"  {f}: comments only: nothing")
        elif not ci and not other:
            lines.append(f"  {f}: only the text rules, which run on every change, or scripts/mutate.py read it: nothing more")
        else:
            lines.append(f"  {f}: {' '.join(ci) or '(no CI target)'}" + (f"; also {' '.join(other)}" if other else ""))
    if sel["broad"]:
        if more > 0:
            lines.append(f"  and {more} more that call for every target")
        if rest:
            lines.append(f"  and {rest} other {'file' if rest == 1 else 'files'}")
    return lines


# ---- commands -------------------------------------------------------------------------------------------------------

def cmd_show(args):
    base = args[0] if args else "origin/devel"
    head = args[1] if len(args) > 1 else "HEAD"
    w = World()
    mb, changed = changed_files(base, head, worktree=len(args) < 2)
    print(f"affected: {base} ({mb[:7]}) to {head}" + (" and the working tree" if len(args) < 2 else "")
          + f": {len(changed)} files")
    sel = w.select(changed, mb, None if len(args) < 2 else head)
    print("\n".join(describe(w, sel)))
    if sel["broad"]:
        print("affected: every target (make check bits check-js)")
    else:
        print("affected: make " + (" ".join(sel["units"]) if sel["units"] else "(nothing)"))
    print(f"affected: {len(sel['mutants'])} mutants")
    for name, why in sel["mutants"]:
        print(f"  {name}  [{why}]")
    if sel["local"]:
        print("affected: on this machine also: make " + " ".join(sel["local"]))


def cmd_self_check(args):
    quiet = "--quiet" in args
    w, bad = World(), []
    say = (lambda *_: None) if quiet else print
    for name, path, before, after, gate, expected in w.mut.MUTANTS:
        try:
            if before not in open(os.path.join(ROOT, path)).read():
                bad.append(f"{name}: its text is not in {path}")
        except OSError:
            bad.append(f"{name}: {path} does not exist")
        if path not in w.gate_files[gate][1]:
            bad.append(f"{name}: its gate {gate.__name__} is not seen to read {path}")
    say(f"affected: the {len(w.mut.MUTANTS)} mutants' texts are in their files, and each gate is seen to read its mutants' file")
    targets = w.ci_units + LOCAL
    for gate, (roots, _) in w.gate_files.items():
        if not roots:
            bad.append(f"gate {gate.__name__}: names no file")
        elif not any(roots <= w.closures[t] for t in targets):
            bad.append(f"gate {gate.__name__}: no make target reads all of {sorted(roots)}")
    say(f"affected: each of the {len(w.gate_files)} gates is a test a make target runs")
    demos = {d for d in w.g.demos if f"demo/{d}.bend" in w.files}
    built = {u[len("build-"):] for u in w.ci_units if u.startswith("build-")}
    listed = set(w.mk.expand("$(DEMOS)").split())
    if built != listed:
        bad.append(f"CI builds {sorted(built)}, DEMOS lists {sorted(listed)}")
    for u in w.ci_units:
        if not w.g.target_roots(u):
            bad.append(f"{u}: its recipe names no file")
    # the Makefile's diff, on planted edits: none, a recipe line, a variable a recipe reads, an export, a comment
    mk_text = open(os.path.join(ROOT, "Makefile")).read()
    targets = w.mk.targets() + w.ci_units
    unit = next(u for u in w.ci_units if u.startswith("check-") and w.mk.rules.get(u, ("", []))[1])
    line = "\t" + w.mk.rules[unit][1][0]
    var = sorted(w.mk.vars_used("laws"))[0]
    var_line = next(l for l in mk_text.split("\n") if re.match(rf"{var}\s*:?=", l))
    planted = [("none", mk_text, lambda h: h == set()),
               (f"{unit}'s recipe", mk_text.replace(line, line + " ", 1), lambda h: h is not None and unit in h and "laws" not in h),
               (f"{var}, which laws reads", mk_text.replace(var_line, var_line + " x", 1), lambda h: h is not None and "laws" in h),
               ("an export", mk_text.replace(w.mk.other[0], w.mk.other[0] + "x", 1) if w.mk.other else None, lambda h: h is None),
               ("a comment", "# planted\n" + mk_text, lambda h: h == set())]
    for what, text, ok in planted:
        if text is not None and not ok(makefile_affected(mk_text, text, targets)):
            bad.append(f"the Makefile's diff: an edit of {what} reaches {makefile_affected(mk_text, text, targets)}")
    if (code("x\n# a\ny", "a.bend") != code("x\n  # b\ny\n", "a.bend") or code("#!/bin/sh\nx", "a.sh") == code("#!/bin/bash\nx", "a.sh")
            or code("x = 1  # a\n", "a.py") != code("x = 1  # b\n", "a.py") or code('s = """\n# a\n"""\n', "a.py") == code('s = """\n# b\n"""\n', "a.py")):
        bad.append("the comment filter: a comment counts as code, or a shebang or a string's # line does not")
    say(f"affected: the Makefile's diff and the comment filter pass {len(planted) + 4} planted edits")
    # the check has teeth: without .bend imports, the gates no longer see most of the files their mutants change
    nw = World(use_imports=False)
    lost = sum(1 for m in nw.mut.MUTANTS if m[1] not in nw.gate_files[m[4]][1])
    if lost == 0:
        bad.append("negative control: without .bend imports every gate still sees its mutants' file, so the check proves nothing")
    say(f"affected: negative control: without .bend imports, {lost} of {len(w.mut.MUTANTS)} mutants' files are not seen by their gates")
    for f in sorted(BY_HAND):
        readers = [t for t, c in w.closures.items() if f in c]
        if f not in w.files or readers:
            bad.append(f"{f}: listed as run by hand, but " + (f"{' '.join(readers)} reads it" if readers else "it is not tracked"))
    unplaced = sorted(f for f in w.files if w.place(f)[0] == "unknown")
    say(f"affected: {len(w.ci_units)} CI targets over {len(demos)} demos; {len(w.universal)} files read by every run"
        + (f"; read by nothing and not documentation (a change to one runs every target): {' '.join(unplaced)}" if unplaced else ""))
    if bad:
        print("affected: self-check FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("affected: self-check passes")


def last_green(head):
    """What is known to pass: a commit with a green run of this workflow on a push or by hand (a pull request's run tests
    a merge, not its head) whose tree is this commit's, else the newest one in this commit's history. A release's merge
    into main has the tree of devel's head, so it finds that commit's run and needs nothing."""
    repo = os.environ["GITHUB_REPOSITORY"]
    out = subprocess.run(["gh", "api", f"repos/{repo}/actions/workflows/ci.yml/runs?status=success&per_page=100", "--jq",
                          '.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch") | .head_sha'],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(f"affected: the green runs could not be read ({out.stderr.strip()[:200]})")
        return None
    tree = git("rev-parse", f"{head}^{{tree}}").strip()
    first = None
    for sha in out.stdout.split():
        t = subprocess.run(["git", "rev-parse", f"{sha}^{{tree}}"], cwd=ROOT, capture_output=True, text=True)
        if t.returncode != 0:
            continue
        if t.stdout.strip() == tree:
            return sha
        if first is None and sha != head and subprocess.run(["git", "merge-base", "--is-ancestor", sha, head], cwd=ROOT).returncode == 0:
            first = sha
    return first


def cmd_ci(args):
    event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
    name = os.environ["GITHUB_EVENT_NAME"]
    head = git("rev-parse", "HEAD").strip()
    full = os.environ.get("FULL") == "true"
    why = "a pull request to main, the weekly run or a run by hand" if full else ""
    if not full and name == "push" and "[full]" in (event.get("head_commit") or {}).get("message", ""):
        full, why = True, "[full] in the commit message"
    base = None
    if not full:
        if name == "pull_request":
            base = event["pull_request"]["base"]["sha"]
        else:
            base = last_green(head)
        if base is None:
            full, why = True, "no green run of this workflow in this branch's history"
    w = World()
    summary = []
    if not full:
        try:
            mb, changed = changed_files(base, head)
        except subprocess.CalledProcessError:
            full, why = True, f"{base[:7]} is not in the history"
        else:
            sel = w.select(changed, mb)
            summary.append(f"Changes since {mb[:7]} ({'the base' if name == 'pull_request' else 'the last green run'}): {len(changed)} files")
            summary += ["```"] + describe(w, sel) + ["```"]
    os.makedirs(os.path.join(ROOT, ".scratch"), exist_ok=True)
    listed = os.path.join(ROOT, ".scratch", "mutants.txt")
    if full:
        targets, js, mutants = "check bits", True, len(w.mut.MUTANTS)
        if os.path.exists(listed):
            os.remove(listed)
        summary.insert(0, f"**Everything** ({why}).")
    else:
        js = "check-js" in sel["units"]
        targets = "check bits" if sel["broad"] else " ".join(u for u in sel["units"] if u != "check-js")
        mutants = len(sel["mutants"])
        open(listed, "w").write("".join(n + "\n" for n, _ in sel["mutants"]))
        summary.insert(0, (f"**Every target** (a file every run reads, or one nothing is seen to read): make {targets}"
                           if sel["broad"] else f"**Selected**: make {targets or '(nothing)'}") + ("; the JavaScript lane" if js else "")
                       + f"; {mutants} mutants.")
        if sel["mutants"]:
            summary += ["", "Mutants:", "```"] + [f"{n}  [{r}]" for n, r in sel["mutants"]] + ["```"]
        if sel["local"]:
            summary += ["", "On the local machine before a release: make " + " ".join(sel["local"])]
    setup = bool(targets or js or mutants)
    with open(os.environ["GITHUB_OUTPUT"], "a") as o:
        o.write(f"full={'true' if full else 'false'}\ntargets={targets}\njs={'true' if js else 'false'}\n"
                f"mutants={mutants}\nsetup={'true' if setup else 'false'}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as o:
            o.write("\n".join(summary) + "\n")
    print("\n".join(summary))


def main():
    cmds = {"show": cmd_show, "self-check": cmd_self_check, "ci": cmd_ci}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
