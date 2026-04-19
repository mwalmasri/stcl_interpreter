# 📘 Schema-Typed Combinatory Logic (STCL) Interpreter

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![CI](https://github.com/YOUR-USERNAME/stcl-interpreter/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/stcl-interpreter/actions)

> A verifiable data workflow language extending combinatory logic with schema-aware typing, formal guarantees, and structure-preserving compilation.

## 📖 Overview
STCL provides a mathematically rigorous foundation for data transformations. By replacing implicit variable binding with pure function composition (`K, S, B, C, I`) and schema-aware types, STCL enables **decidable syntactic equivalence**, **strong normalization**, and **categorical semantics** for data pipelines.

Based on the paper: *"Schema-Typed Combinatory Logic: A Mathematical Foundation for Verifiable Data Workflows"* by Walid AlMasri.

## ✨ Core Features
| Feature | Description |
|---------|-------------|
| 🔒 **Formal Guarantees** | Proved subject reduction, strong normalization & confluence (Church-Rosser) |
| 📐 **Categorical Semantics** | Symmetric Monoidal Closed Category (SMCC) with schema objects & functorial interpretation |
| 🔍 **Decidable Equivalence** | Syntactic term equivalence checking with refined worst-case/average-case bounds |
| 🧩 **Extensible Primitives** | Unary & binary operations with schema-typed signatures and graceful error handling |
| ⚡ **Production Interpreter** | Leftmost-outermost reduction, schema type checker, parser & benchmark suite |

## 🚀 Quick Start

### Installation
```bash
pip install -e ".[dev]"
