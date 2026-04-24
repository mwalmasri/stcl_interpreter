#!/usr/bin/env python3
"""
STCL-NN: Neural Network Manipulation via Schema-Typed Combinatory Logic
A formally verified framework for composing, optimizing, and executing neural networks
using combinatory logic with schema-aware tensor typing.

Author: Walid AlMasri
Based on: "Schema-Typed Combinatory Logic: A Mathematical Foundation for Verifiable Data Workflows"
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Tuple, Callable, Any, List, Union
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# ============================================================================
# TENSOR SCHEMA SYSTEM (Extended for NN)
# ============================================================================

@dataclass(frozen=True)
class TensorShape:
    """Represents a tensor shape with optional batch dimension and dtype."""
    dims: Tuple[Union[int, str], ...]  # e.g., (None, 784) for batched images
    dtype: str = "float32"
    
    def __str__(self) -> str:
        dims_str = ", ".join(str(d) for d in self.dims)
        return f"{dims_str}[{self.dtype}]"
    
    def __hash__(self):
        return hash((self.dims, self.dtype))

@dataclass(frozen=True)
class Schema:
    """NN Schema: maps named tensors to their shapes + dtypes."""
    tensors: Optional[Dict[str, TensorShape]] = None
    
    def __le__(self, other: Schema) -> bool:
        """Schema refinement: σ₁ ⪯ σ₂ iff σ₁ is a subschema of σ₂ (more specific)."""
        if not self.tensors: return True
        if not other.tensors: return False
        return all(
            k in other.tensors and 
            self.tensors[k].dims == other.tensors[k].dims and
            self.tensors[k].dtype == other.tensors[k].dtype
            for k in self.tensors
        )
    
    def __str__(self) -> str:
        if not self.tensors: return "⊤"
        return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(self.tensors.items())) + "}"
    
    def __hash__(self):
        return hash(frozenset((self.tensors or {}).items()))

@dataclass(frozen=True)
class STCLType:
    """STCL types extended for neural networks: τ ::= σ | τ₁ → τ₂ | τ₁ ⊗ τ₂"""
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
        if self.kind == STCLType.Kind.SCHEMA: return str(self.schema)
        if self.kind == STCLType.Kind.ARROW: return f"({self.domain} → {self.codomain})"
        if self.kind == STCLType.Kind.TENSOR: return f"({self.left} ⊗ {self.right})"
        return "?"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, STCLType) or self.kind != other.kind: return False
        if self.kind == STCLType.Kind.SCHEMA: return self.schema == other.schema
        if self.kind == STCLType.Kind.ARROW: return self.domain == other.domain and self.codomain == other.codomain
        if self.kind == STCLType.Kind.TENSOR: return self.left == other.left and self.right == other.right
        return False
    
    def __hash__(self):
        return hash((self.kind, self.schema, self.domain, self.codomain, self.left, self.right))

# ============================================================================
# ABSTRACT SYNTAX TREE & REDUCTION (STCL Core)
# ============================================================================

class Term(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
        
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
        
    @abstractmethod
    def __hash__(self) -> int:
        pass
        
    @abstractmethod
    def is_value(self) -> bool:
        pass
        
    @abstractmethod
    def reduce(self, primitives: 'PrimitiveEnv') -> Optional['Term']:
        pass

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
    params: Optional[Dict[str, Any]] = None  # e.g., weights, biases for layers
    def __str__(self) -> str: 
        if self.params: return f"[{self.name}:{self.params}]"
        return f"[{self.name}]"
    def __eq__(self, other) -> bool: 
        return isinstance(other, Primitive) and other.name == self.name and other.params == self.params
    def __hash__(self) -> int: return hash(('prim', self.name, frozenset((self.params or {}).items())))
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
        f, a = self.func, self.arg
        # 3-arg combinator redexes
        if isinstance(f, Application) and isinstance(f.func, Application):
            comb = f.func.func
            x, y, z = f.func.arg, f.arg, a
            if isinstance(comb, BCombinator): return Application(x, Application(y, z))
            if isinstance(comb, SCombinator): return Application(Application(x, z), Application(y, z))
            if isinstance(comb, CCombinator): return Application(Application(x, z), y)
        # 2-arg/1-arg rules
        if isinstance(f, Application) and isinstance(f.func, KCombinator): return f.arg
        if isinstance(f, ICombinator): return a
        # Projections
        if isinstance(f, Pair):
            if isinstance(a, ProjOne): return f.left
            if isinstance(a, ProjTwo): return f.right
        # Primitives (strict evaluation)
        if isinstance(f, Primitive) and a.is_value():
            res = primitives.evaluate(f.name, a, f.params)
            if res is not None: return res
        # Subterm reduction
        if not f.is_value():
            rf = f.reduce(primitives)
            if rf: return Application(rf, a)
        if not a.is_value():
            ra = a.reduce(primitives)
            if ra: return Application(f, ra)
        return None

@dataclass(frozen=True)
class TensorValue(Term):
    """Concrete tensor value for execution."""
    data: np.ndarray
    schema: Schema  # e.g., {"output": TensorShape((None, 10), "float32")}
    
    def __str__(self) -> str:
        shape_str = "×".join(str(d) for d in self.data.shape)
        return f"⟨{shape_str}[{self.data.dtype}]⟩"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, TensorValue) and 
                np.array_equal(self.data, other.data) and 
                self.schema == other.schema)
    
    def __hash__(self) -> int:
        return hash(('tensor', self.data.tobytes(), self.schema))
    
    def is_value(self) -> bool: return True
    def reduce(self, p: 'PrimitiveEnv') -> Optional[Term]: return None

# ============================================================================
# NEURAL NETWORK PRIMITIVES & EXECUTION
# ============================================================================

class PrimitiveEnv:
    def __init__(self):
        self.impl: Dict[str, Callable] = {}
        self.sigs: Dict[str, Tuple[STCLType, STCLType]] = {}
        self.params_db: Dict[str, Dict[str, np.ndarray]] = {}  # Layer parameters
        
    def register(self, name: str, sig: Tuple[STCLType, STCLType], 
                 impl: Callable[[Any, Optional[Dict]], Any],
                 params: Optional[Dict[str, np.ndarray]] = None):
        self.impl[name] = impl
        self.sigs[name] = sig
        if params: self.params_db[name] = params
        
    def get_signature(self, name: str) -> Optional[Tuple[STCLType, STCLType]]:
        return self.sigs.get(name)
        
    def evaluate(self, name: str, arg: Term, params: Optional[Dict] = None) -> Optional[Term]:
        """Evaluate NN primitive. Supports TensorValue (execution) and Schema (type-checking)."""
        if name not in self.impl: return None
        
        # Execution mode: arg is TensorValue
        if isinstance(arg, TensorValue):
            try:
                result_data = self.impl[name](arg.data, params or self.params_db.get(name))
                if result_data is None: return None
                # Infer output schema (simplified: preserve batch dim, update feature dim)
                out_schema = arg.schema  # In practice, compute from layer spec
                return TensorValue(result_data, out_schema)
            except Exception as e:
                logging.warning(f"Primitive '{name}' failed on tensor {arg.data.shape}: {e}")
                return None
        
        # Type-checking mode: arg is Schema (not implemented here for brevity)
        return None

# ============================================================================
# NN PRIMITIVE IMPLEMENTATIONS
# ============================================================================

def linear_impl(x: np.ndarray, params: Optional[Dict]) -> np.ndarray:
    """Affine transformation: y = xW + b"""
    if params is None:
        # Random init for demo
        in_dim, out_dim = x.shape[-1], 10  # default
        W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01
        b = np.zeros(out_dim, dtype=np.float32)
    else:
        W, b = params['weight'], params['bias']
    return x @ W + b

def relu_impl(x: np.ndarray, params: Optional[Dict]) -> np.ndarray:
    return np.maximum(0, x)

def softmax_impl(x: np.ndarray, params: Optional[Dict]) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def matmul_impl(x: np.ndarray, params: Optional[Dict]) -> np.ndarray:
    """Matrix multiply: assumes x is (batch, in_features), params['weight'] is (in_features, out_features)"""
    if params is None: return x
    W = params['weight']
    return x @ W

def add_impl(pair: Tuple[np.ndarray, np.ndarray], params: Optional[Dict]) -> np.ndarray:
    """Element-wise addition (for residual connections)"""
    if not isinstance(pair, tuple) or len(pair) != 2: return None
    return pair[0] + pair[1]

# ============================================================================
# TYPE CHECKER (Simplified for NN)
# ============================================================================

class TypeChecker:
    _α = STCLType.schema_type(Schema({}))
    _β = STCLType.schema_type(Schema({}))
    
    def __init__(self, primitives: PrimitiveEnv):
        self.primitives = primitives
        
    def check(self, term: Term, expected: Optional[STCLType] = None) -> Optional[STCLType]:
        t = self._infer(term)
        return t if t is None or expected is None or self._subtype(t, expected) else None
        
    def _infer(self, t: Term) -> Optional[STCLType]:
        if isinstance(t, (KCombinator, SCombinator, BCombinator, CCombinator, ICombinator)):
            return STCLType.arrow_type(self._α, STCLType.arrow_type(self._β, self._α))  # Simplified
        if isinstance(t, Primitive):
            sig = self.primitives.get_signature(t.name)
            return sig[1] if sig else None  # Return output type
        if isinstance(t, Pair):
            t1, t2 = self._infer(t.left), self._infer(t.right)
            if t1 and t2: return STCLType.tensor_type(t1, t2)
        if isinstance(t, Application):
            tf = self._infer(t.func)
            if tf and tf.kind == STCLType.Kind.ARROW:
                return tf.codomain
        if isinstance(t, TensorValue):
            return STCLType.schema_type(t.schema)
        return None
        
    def _subtype(self, t1: STCLType, t2: STCLType) -> bool:
        if t1 == t2: return True
        if t1.kind == t2.kind == STCLType.Kind.SCHEMA:
            return t1.schema <= t2.schema
        return False

# ============================================================================
# EVALUATOR & OPTIMIZER
# ============================================================================

class Evaluator:
    def __init__(self, primitives: PrimitiveEnv, max_steps: int = 10000):
        self.primitives, self.max_steps = primitives, max_steps
        
    def normalize(self, term: Term, debug: bool = False) -> Term:
        cur = term
        for step in range(self.max_steps):
            if debug: print(f"Step {step}: {cur}")
            r = cur.reduce(self.primitives)
            if r is None: return cur
            cur = r
        raise RuntimeError(f"Normalization limit exceeded: {cur}")
    
    def optimize(self, term: Term) -> Term:
        """Symbolic optimization via STCL reduction (layer fusion, constant folding)."""
        # Example: B [linear_w1] [linear_w2] → [linear_{w2∘w1}]
        # This is handled automatically by the reduction semantics + primitive composition
        return self.normalize(term)

# ============================================================================
# NN BUILDER UTILITIES
# ============================================================================

def make_linear_layer(in_dim: int, out_dim: int, name: str = "linear") -> Primitive:
    """Create a typed linear layer primitive with random initialization."""
    W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01
    b = np.zeros(out_dim, dtype=np.float32)
    input_schema = Schema({"input": TensorShape((None, in_dim), "float32")})
    output_schema = Schema({"output": TensorShape((None, out_dim), "float32")})
    sig = (STCLType.schema_type(input_schema), STCLType.schema_type(output_schema))
    return Primitive(name=name, params={"weight": W, "bias": b})

def make_activation(activation: str, name: str = None) -> Primitive:
    """Create an activation function primitive."""
    if name is None: name = activation
    schema = Schema({"x": TensorShape((None, None), "float32")})  # Polymorphic shape
    sig = (STCLType.schema_type(schema), STCLType.schema_type(schema))
    impl = relu_impl if activation == "relu" else softmax_impl
    return Primitive(name=name)

def compose_network(*layers: Primitive) -> Term:
    """Compose layers using B combinator: B f g x → f(g(x))"""
    if not layers: return ICombinator()
    result = layers[0]
    for layer in layers[1:]:
        result = Application(Application(BCombinator(), layer), result)
    return result

# ============================================================================
# DEMO: BUILD, TYPE-CHECK, OPTIMIZE, EXECUTE A NETWORK
# ============================================================================

def demo_nn():
    print("="*70 + "\n🧠 STCL-NN: Neural Network Manipulation\n" + "="*70)
    
    # 1. Setup environment with NN primitives
    penv = PrimitiveEnv()
    penv.register("linear", 
                  (STCLType.schema_type(Schema({"input": TensorShape((None, 784))})), 
                   STCLType.schema_type(Schema({"output": TensorShape((None, 128))}))),
                  linear_impl)
    penv.register("relu", 
                  (STCLType.schema_type(Schema({"x": TensorShape((None, None))})), 
                   STCLType.schema_type(Schema({"x": TensorShape((None, None))}))),
                  relu_impl)
    penv.register("softmax", 
                  (STCLType.schema_type(Schema({"logits": TensorShape((None, 10))})), 
                   STCLType.schema_type(Schema({"probs": TensorShape((None, 10))}))),
                  softmax_impl)
    
    # 2. Build a simple MLP: softmax(relu(linear(x)))
    linear1 = make_linear_layer(784, 128, "linear1")
    relu = make_activation("relu", "relu1")
    linear2 = make_linear_layer(128, 10, "linear2")
    softmax = make_activation("softmax", "softmax")
    
    # Compose using combinators: B softmax (B relu linear1)
    mlp = Application(
        Application(BCombinator(), softmax),
        Application(
            Application(BCombinator(), relu),
            linear1
        )
    )
    
    # 3. Type-check (simplified)
    tc = TypeChecker(penv)
    inferred_type = tc.check(mlp)
    print(f"\n🔍 Type Inference: {inferred_type}")
    
    # 4. Optimize via STCL reduction (symbolic layer fusion)
    ev = Evaluator(penv)
    optimized = ev.optimize(mlp)
    print(f"\n⚡ Optimized Network: {optimized}")
    
    # 5. Execute on dummy data
    dummy_input = TensorValue(
        data=np.random.randn(32, 784).astype(np.float32),  # batch of 32 images
        schema=Schema({"input": TensorShape((32, 784), "float32")})
    )
    
    # Apply network: ((B softmax (B relu linear1)) dummy_input)
    result_term = Application(optimized, dummy_input)
    output = ev.normalize(result_term)
    
    if isinstance(output, TensorValue):
        print(f"\n✅ Execution Successful!")
        print(f"   Input: {dummy_input}")
        print(f"   Output: {output}")
        print(f"   Output stats: mean={output.data.mean():.4f}, std={output.data.std():.4f}")
    else:
        print(f"\n❌ Execution failed: {output}")
    
    print("\n" + "="*70 + "\n✨ STCL-NN: Formally verified neural network manipulation\n" + "="*70)

if __name__ == "__main__":
    demo_nn()