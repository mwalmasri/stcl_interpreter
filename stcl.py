#!/usr/bin/env python3
"""
Schema-Typed Combinatory Logic (STCL) Interpreter - FULL VERSION
A verifiable data workflow language based on combinatory logic with schema-aware typing.
Includes: Core Logic, Type System, Parser, and Basic Arithmetic Library.

Author: Walid AlMasri
Based on: "Schema-Typed Combinatory Logic: A Mathematical Foundation for Verifiable Data Workflows"
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Tuple, Callable, Any, List
import logging
import sys

# Configure logging for primitive error reporting
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# ============================================================================
# CORE TYPE SYSTEM
# ============================================================================

@dataclass(frozen=True)
class Schema:
    """A schema is a finite set of typed attributes: {A1:τ1, ..., Ak:τk}"""
    attributes: Optional[Dict[str, 'STCLType']] = None
    
    def __le__(self, other: Schema) -> bool:
        """Schema refinement: σ₁ ⪯ σ₂ iff σ₁ is a subschema of σ₂."""
        if not self.attributes:
            return True
        if not other.attributes:
            return False
        return all(k in other.attributes and self.attributes[k] == other.attributes[k] 
                  for k in self.attributes)
    
    def __str__(self) -> str:
        if not self.attributes:
            return "⊤"
        return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(self.attributes.items())) + "}"
    
    def __hash__(self):
        return hash(frozenset((self.attributes or {}).items()))


@dataclass(frozen=True)
class STCLType:
    """STCL types: τ ::= σ | τ₁ → τ₂ | τ₁ ⊗ τ₂"""
    class Kind(Enum):
        SCHEMA = auto()
        ARROW = auto()
        TENSOR = auto()
        
    kind: Kind
    schema: Optional[Schema] = None
    domain: Optional['STCLType'] = None
    codomain: Optional['STCLType'] = None
    left: Optional['STCLType'] = None
    right: Optional['STCLType'] = None
    
    @staticmethod
    def schema_type(s: Schema) -> 'STCLType':
        return STCLType(kind=STCLType.Kind.SCHEMA, schema=s)

    @staticmethod
    def arrow_type(dom: 'STCLType', cod: 'STCLType') -> 'STCLType':
        return STCLType(kind=STCLType.Kind.ARROW, domain=dom, codomain=cod)

    @staticmethod
    def tensor_type(l: 'STCLType', r: 'STCLType') -> 'STCLType':
        return STCLType(kind=STCLType.Kind.TENSOR, left=l, right=r)
    
    def __str__(self) -> str:
        if self.kind == STCLType.Kind.SCHEMA:
            return str(self.schema)
        if self.kind == STCLType.Kind.ARROW:
            return f"({self.domain} → {self.codomain})"
        if self.kind == STCLType.Kind.TENSOR:
            return f"({self.left} ⊗ {self.right})"
        return "?"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, STCLType) or self.kind != other.kind:
            return False
        if self.kind == STCLType.Kind.SCHEMA:
            return self.schema == other.schema
        if self.kind == STCLType.Kind.ARROW:
            return self.domain == other.domain and self.codomain == other.codomain
        if self.kind == STCLType.Kind.TENSOR:
            return self.left == other.left and self.right == other.right
        return False
    
    def __hash__(self):
        return hash((self.kind, self.schema, self.domain, self.codomain, self.left, self.right))


# ============================================================================
# ABSTRACT SYNTAX TREE & REDUCTION
# ============================================================================

class Term(ABC):
    @abstractmethod
    def __str__(self) -> str: pass
        
    @abstractmethod
    def __eq__(self, other) -> bool: pass
        
    @abstractmethod
    def __hash__(self) -> int: pass
        
    @abstractmethod
    def is_value(self) -> bool: pass
        
    @abstractmethod
    def reduce(self, primitives: 'PrimitiveEnv') -> Optional[Term]: pass


@dataclass(frozen=True)
class KCombinator(Term):
    def __str__(self) -> str: return "K"
    def __eq__(self, other) -> bool: return isinstance(other, KCombinator)
    def __hash__(self) -> int: return hash('K')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class SCombinator(Term):
    def __str__(self) -> str: return "S"
    def __eq__(self, other) -> bool: return isinstance(other, SCombinator)
    def __hash__(self) -> int: return hash('S')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class BCombinator(Term):
    def __str__(self) -> str: return "B"
    def __eq__(self, other) -> bool: return isinstance(other, BCombinator)
    def __hash__(self) -> int: return hash('B')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class CCombinator(Term):
    def __str__(self) -> str: return "C"
    def __eq__(self, other) -> bool: return isinstance(other, CCombinator)
    def __hash__(self) -> int: return hash('C')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class ICombinator(Term):
    def __str__(self) -> str: return "I"
    def __eq__(self, other) -> bool: return isinstance(other, ICombinator)
    def __hash__(self) -> int: return hash('I')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class Primitive(Term):
    name: str
    def __str__(self) -> str: return f"[{self.name}]"
    def __eq__(self, other) -> bool: return isinstance(other, Primitive) and other.name == self.name
    def __hash__(self) -> int: return hash(('prim', self.name))
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class ProjOne(Term):
    def __str__(self) -> str: return "π₁"
    def __eq__(self, other) -> bool: return isinstance(other, ProjOne)
    def __hash__(self) -> int: return hash('π₁')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class ProjTwo(Term):
    def __str__(self) -> str: return "π₂"
    def __eq__(self, other) -> bool: return isinstance(other, ProjTwo)
    def __hash__(self) -> int: return hash('π₂')
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

@dataclass(frozen=True)
class Pair(Term):
    left: Term
    right: Term
    def __str__(self) -> str: return f"⟨{self.left}, {self.right}⟩"
    def __eq__(self, other) -> bool: return isinstance(other, Pair) and self.left == other.left and self.right == other.right
    def __hash__(self) -> int: return hash(('pair', self.left, self.right))
    def is_value(self) -> bool: return self.left.is_value() and self.right.is_value()
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]:
        if not self.left.is_value():
            r = self.left.reduce(p)
            if r: return Pair(r, self.right)
        if not self.right.is_value():
            r = self.right.reduce(p)
            if r: return Pair(self.left, r)
        return None

@dataclass(frozen=True)
class Application(Term):
    func: Term
    arg: Term
    def __str__(self) -> str: return f"({self.func} {self.arg})"
    def __eq__(self, other) -> bool: return isinstance(other, Application) and self.func == other.func and self.arg == other.arg
    def __hash__(self) -> int: return hash(('app', self.func, self.arg))
    def is_value(self) -> bool: return False

    def reduce(self, primitives: 'PrimitiveEnv') -> Optional[Term]:
        """Strict leftmost-outermost reduction with safe argument extraction."""
        f, a = self.func, self.arg
        
        # Check for 3-arg combinator redexes: ((C x) y) z
        if isinstance(f, Application) and isinstance(f.func, Application):
            comb = f.func.func
            x = f.func.arg  # First argument
            y = f.arg       # Second argument
            z = a           # Third argument
            
            if isinstance(comb, BCombinator):
                return Application(x, Application(y, z))
            if isinstance(comb, SCombinator):
                return Application(Application(x, z), Application(y, z))
            if isinstance(comb, CCombinator):
                return Application(Application(x, z), y)
            
        # 2-arg/1-arg rules
        if isinstance(f, Application) and isinstance(f.func, KCombinator):
            return f.arg  # K x y → x
        if isinstance(f, ICombinator):
            return a  # I x → x
            
        # Projections
        if isinstance(f, Pair):
            if isinstance(a, ProjOne): return f.left
            if isinstance(a, ProjTwo): return f.right
            
        # Primitives (strict evaluation: requires value argument)
        if isinstance(f, Primitive) and a.is_value():
            res = primitives.evaluate(f.name, a)
            if res is not None: return res

        # Subterm reduction (left-to-right)
        if not f.is_value():
            rf = f.reduce(primitives)
            if rf: return Application(rf, a)
        if not a.is_value():
            ra = a.reduce(primitives)
            if ra: return Application(f, ra)
            
        return None  # Normal form reached

@dataclass(frozen=True)
class ConcreteValue(Term):
    value: Any
    def __str__(self) -> str:
        return repr(self.value)
    def __eq__(self, other) -> bool:
        return isinstance(other, ConcreteValue) and self.value == other.value
    def __hash__(self) -> int:
        return hash(('val', self.value))
    def is_value(self) -> bool:
        return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]:
        return None


# ============================================================================
# PRIMITIVE ENV, EVALUATOR, TYPE CHECKER, PARSER
# ============================================================================

class PrimitiveEnv:
    def __init__(self):
        self.impl: Dict[str, Callable] = {}
        self.sigs: Dict[str, Tuple[STCLType, STCLType]] = {}
        
    def register(self, name: str, sig: Tuple[STCLType, STCLType], impl: Callable[[Any], Any]):
        self.impl[name], self.sigs[name] = impl, sig
        
    def get_signature(self, name: str) -> Optional[Tuple[STCLType, STCLType]]:
        return self.sigs.get(name)
        
    def evaluate(self, name: str, arg: Term) -> Optional[Term]:
        """Evaluate a primitive. Supports both Unary (ConcreteValue) and Binary (Pair) operations."""
        
        # --- CASE 1: Binary Operations (e.g., [add] ⟨2, 3⟩) ---
        if isinstance(arg, Pair) and arg.is_value():
            if name not in self.impl: return None
            # Ensure both sides of the pair are concrete values
            if isinstance(arg.left, ConcreteValue) and isinstance(arg.right, ConcreteValue):
                val_tuple = (arg.left.value, arg.right.value)
                try:
                    r = self.impl[name](val_tuple)
                    return ConcreteValue(r) if r is not None else None
                except Exception as e:
                    logging.warning(f"Primitive '{name}' failed on pair {val_tuple}: {e}")
                    return None
        
        # --- CASE 2: Unary Operations (e.g., [square] 5) ---
        if isinstance(arg, ConcreteValue):
            if name not in self.impl: return None
            try:
                r = self.impl[name](arg.value)
                return ConcreteValue(r) if r is not None else None
            except Exception as e:
                logging.warning(f"Primitive '{name}' failed on {arg.value}: {e}")
                return None
        return None


class TypeChecker:
    _α = STCLType.schema_type(Schema({}))
    _β = STCLType.schema_type(Schema({}))
    _γ = STCLType.schema_type(Schema({}))
    
    def __init__(self, primitives: PrimitiveEnv):
        self.primitives = primitives
        
    def check(self, term: Term, expected: Optional[STCLType] = None) -> Optional[STCLType]:
        t = self._infer(term)
        return t if t is None or expected is None or self._subtype(t, expected) else None
        
    def _infer(self, t: Term) -> Optional[STCLType]:
        if isinstance(t, KCombinator):
            return STCLType.arrow_type(self._α, STCLType.arrow_type(self._β, self._α))
        if isinstance(t, SCombinator):
            abc = STCLType.arrow_type(self._α, STCLType.arrow_type(self._β, self._γ))
            return STCLType.arrow_type(abc, STCLType.arrow_type(STCLType.arrow_type(self._α,self._β), STCLType.arrow_type(self._α,self._γ)))
        if isinstance(t, BCombinator):
            return STCLType.arrow_type(STCLType.arrow_type(self._β,self._γ), STCLType.arrow_type(STCLType.arrow_type(self._α,self._β), STCLType.arrow_type(self._α,self._γ)))
        if isinstance(t, CCombinator):
            abc = STCLType.arrow_type(self._α, STCLType.arrow_type(self._β, self._γ))
            return STCLType.arrow_type(abc, STCLType.arrow_type(self._β, STCLType.arrow_type(self._α, self._γ)))
        if isinstance(t, ICombinator):
            return STCLType.arrow_type(self._α, self._α)
        if isinstance(t, Primitive):
            sig = self.primitives.get_signature(t.name)
            return STCLType.arrow_type(*sig) if sig else None
        if isinstance(t, Pair):
            t1, t2 = self._infer(t.left), self._infer(t.right)
            return STCLType.tensor_type(t1, t2) if t1 and t2 and t1.kind == t2.kind == STCLType.Kind.SCHEMA else None
        if isinstance(t, Application):
            tf, ta = self._infer(t.func), self._infer(t.arg)
            if tf and ta and tf.kind == STCLType.Kind.ARROW and self._subtype(ta, tf.domain):
                return tf.codomain
            return None
        if isinstance(t, ConcreteValue):
            return STCLType.schema_type(Schema({"value": STCLType.schema_type(Schema({}))}))
        return None
        
    def _subtype(self, t1: STCLType, t2: STCLType) -> bool:
        if t1 == t2: return True
        if t1.kind == t2.kind == STCLType.Kind.SCHEMA:
            return t1.schema <= t2.schema
        if t1.kind == t2.kind == STCLType.Kind.ARROW:
            return self._subtype(t2.domain, t1.domain) and self._subtype(t1.codomain, t2.codomain)
        if t1.kind == t2.kind == STCLType.Kind.TENSOR:
            return self._subtype(t1.left, t2.left) and self._subtype(t1.right, t2.right)
        return False


class Evaluator:
    def __init__(self, primitives: PrimitiveEnv, max_steps: int = 10000):
        self.primitives, self.max_steps = primitives, max_steps
        
    def normalize(self, term: Term, debug: bool = False) -> Term:
        cur = term
        for step in range(self.max_steps):
            if debug:
                print(f"Step {step}: {cur}")
            r = cur.reduce(self.primitives)
            if r is None:
                return cur
            cur = r
        raise RuntimeError(f"Normalization limit exceeded after {self.max_steps} steps: {cur}")


class Parser:
    COMBS = {'K','S','B','C','I'}
    
    def __init__(self, primitives: PrimitiveEnv):
        self.primitives = primitives
        self.tokens: List[str] = []
        self.pos: int = 0
        
    def parse(self, src: str) -> Optional[Term]:
        self.tokens = self._tokenize(src)
        self.pos = 0
        r = self._parse_term()
        return r if r and self.pos == len(self.tokens) else None
        
    def _tokenize(self, s: str) -> List[str]:
        toks: List[str] = []
        i = 0
        while i < len(s):
            if s[i].isspace(): i += 1; continue
            if s[i] in 'KSBIC(),': toks.append(s[i]); i += 1; continue
            if s[i] == '⟨': toks.append('⟨'); i += 1; continue
            if s[i:i+2] in ('π₁', 'π₂'): toks.append(s[i:i+2]); i += 2; continue
            if s[i] == '[':
                j = s.find(']', i)
                toks.append(s[i:j+1] if j != -1 else s[i:])
                i = (j + 1 if j != -1 else len(s)); continue
            if s[i].isalpha() or s[i].isdigit() or s[i] in '_-':
                j = i
                while j < len(s) and (s[j].isalnum() or s[j] in '_-.'): j += 1
                toks.append(s[i:j]); i = j; continue
            if s[i] == '"':
                j = i + 1
                while j < len(s) and s[j] != '"': j += 2 if s[j] == '\\' else 1
                if j < len(s): toks.append(s[i:j+1]); i = j + 1
                else: i = len(s)
                continue
            i += 1
        return toks
        
    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
        
    def _consume(self, exp: Optional[str] = None) -> Optional[str]:
        t = self._peek()
        if exp and t != exp: return None
        if t: self.pos += 1
        return t
        
    def _parse_term(self) -> Optional[Term]:
        terms: List[Term] = []
        while (a := self._parse_atom()): terms.append(a)
        if not terms: return None
        r = terms[0]
        for t in terms[1:]:
            r = Application(r, t)  # Left-associative
        return r
        
    def _parse_atom(self) -> Optional[Term]:
        t = self._peek()
        if t in self.COMBS:
            self._consume()
            return {'K': KCombinator(), 'S': SCombinator(), 'B': BCombinator(), 'C': CCombinator(), 'I': ICombinator()}[t]
        if t == 'π₁': self._consume(); return ProjOne()
        if t == 'π₂': self._consume(); return ProjTwo()
        if t == '⟨':
            self._consume()
            l = self._parse_term()
            if not l or not self._consume(','): return None
            r = self._parse_term()
            if not r or not self._consume('⟩'): return None
            return Pair(l, r)
        if t and t.startswith('[') and t.endswith(']'):
            self._consume()
            return Primitive(t[1:-1])
        if t == '(':
            self._consume()
            e = self._parse_term()
            if not e or not self._consume(')'): return None
            return e
        if t:
            for conv in [int, float]:
                try:
                    v = conv(t)
                    self._consume()
                    return ConcreteValue(v)
                except (ValueError, TypeError): pass
            if t.startswith('"') and t.endswith('"'):
                self._consume()
                return ConcreteValue(t[1:-1])
        return None


# ============================================================================
# DEMO & TESTING
# ============================================================================

def test_combinators(penv: PrimitiveEnv, ev: Evaluator):
    print("\n🧪 Combinator Tests:")
    
    k_test = Application(Application(KCombinator(), ConcreteValue(42)), ConcreteValue(99))
    result = ev.normalize(k_test)
    assert isinstance(result, ConcreteValue) and result.value == 42, f"K failed: {result}"
    print("  ✓ K x y → x")
    
    b_test = Application(Application(Application(BCombinator(), Primitive("double")), Primitive("square")), ConcreteValue(3))
    result = ev.normalize(b_test)
    assert isinstance(result, ConcreteValue) and result.value == 18, f"B failed: {result}"
    print("  ✓ B [double] [square] 3 → 18")
    
    pair_test = Application(Pair(ConcreteValue("hello"), ConcreteValue("world")), ProjOne())
    result = ev.normalize(pair_test)
    assert isinstance(result, ConcreteValue) and result.value == "hello"
    print("  ✓ ⟨a,b⟩ π₁ → a")
    
    print("  ✅ All combinator tests passed!")

def test_arithmetic(penv: PrimitiveEnv, ev: Evaluator):
    print("\n🧪 Arithmetic Tests:")
    
    # Unary: [square] 5 → 25
    sq_test = Application(Primitive("square"), ConcreteValue(5))
    res = ev.normalize(sq_test)
    assert isinstance(res, ConcreteValue) and res.value == 25, f"Square failed: {res}"
    print("  ✓ [square] 5 → 25")
    
    # Binary: [add] ⟨2, 3⟩ → 5
    pair_arg = Pair(ConcreteValue(2), ConcreteValue(3))
    add_test = Application(Primitive("add"), pair_arg)
    res = ev.normalize(add_test)
    assert isinstance(res, ConcreteValue) and res.value == 5, f"Add failed: {res}"
    print("  ✓ [add] ⟨2, 3⟩ → 5")
    
    # Complex: [mul] ⟨[add] ⟨1, 2⟩, 3⟩ → (1+2)*3 = 9
    inner_add = Application(Primitive("add"), Pair(ConcreteValue(1), ConcreteValue(2)))
    mul_arg = Pair(inner_add, ConcreteValue(3))
    mul_test = Application(Primitive("mul"), mul_arg)
    res = ev.normalize(mul_test)
    # Note: Inputs are ints, result is int. 
    # (1+2)*3 = 9.
    assert isinstance(res, ConcreteValue) and res.value == 9, f"Mul complex failed: {res}"
    print("  ✓ [mul] ⟨[add] ⟨1, 2⟩, 3⟩ → 9")
    
    print("  ✅ All arithmetic tests passed!")


def demo(debug: bool = False):
    print("=" * 70 + "\nSchema-Typed Combinatory Logic (STCL) Interpreter\n" + "=" * 70)
    penv = PrimitiveEnv()
    
    # Define Schemas
    num_schema = Schema({})
    
    # --- REGISTER PRIMITIVES ---
    # Unary
    penv.register("square", (num_schema, num_schema), lambda v: v*v if isinstance(v, (int, float)) else None)
    penv.register("double", (num_schema, num_schema), lambda v: v*2 if isinstance(v, (int, float)) else None)
    
    # Binary (takes Tuple (x,y) from Pair)
    penv.register("add", (num_schema, num_schema), lambda p: p[0] + p[1])
    penv.register("sub", (num_schema, num_schema), lambda p: p[0] - p[1])
    penv.register("mul", (num_schema, num_schema), lambda p: p[0] * p[1])
    penv.register("div", (num_schema, num_schema), lambda p: p[0] / p[1] if p[1] != 0 else None)
    
    tc, ev, parser = TypeChecker(penv), Evaluator(penv), Parser(penv)
    
    test_combinators(penv, ev)
    test_arithmetic(penv, ev)
    
    if debug:
        print(f"\n🔍 Debug mode: Tracing [add] ⟨2, 3⟩")
        term = Application(Primitive("add"), Pair(ConcreteValue(2), ConcreteValue(3)))
        ev_debug = Evaluator(penv, max_steps=10)
        cur = term
        step = 0
        while True:
            print(f"  Step {step}: {cur}")
            nxt = cur.reduce(penv)
            if nxt is None:
                print(f"  → Normal form reached")
                break
            cur = nxt
            step += 1
            if step > 10: break
    else:
        print("\n[1] Parsing & Evaluation Examples:")
        for src in ["K 1 2", "[add] ⟨10, 20⟩", "[mul] ⟨[add] ⟨2, 3⟩, 4⟩"]:
            parsed = parser.parse(src)
            print(f"    '{src}' → {ev.normalize(parsed) if parsed else 'Parse failed'}")
            
    print("\n" + "=" * 70 + "\nSTCL interpreter ready.\n" + "=" * 70)


if __name__ == "__main__":
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    demo(debug=debug_mode)
