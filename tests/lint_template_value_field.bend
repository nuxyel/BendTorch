# Fixture for bend-lint's performance/template_value_field (`bend-lint-rules@0.2.2.0`, `make lint`): a record of
# functions with a value field computed from templates is reported; the same field as a closure, and a record of plain
# values, are not.
import Base

type R is Type:
  R{f: U32 -> U32, n: Unit -> U32}

type E is Type:
  E{f: U32 -> U32, n: U32}

def R.n(r: R) -> U32:
  match r:
    case R{f, n}: n(Unit{})

def E.n(r: E) -> U32:
  match r:
    case E{f, n}: n

def eager(~a: E, ~b: E) -> E:
  # bend-lint: expect-next bend-lint-rules/performance/template_value_field@information -- the eager count of BendTorch's M9, kept as the rule's test
  E{x => x, (E.n(a) + E.n(b) : U32)}

def lazy(~a: R, ~b: R) -> R:
  R{x => x, u => (R.n(a) + R.n(b) : U32)}

def plain(~a: R) -> U32 & U32:
  Tuple{R.n(a), 0}
