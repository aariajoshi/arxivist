"""
quantum_cryptanalysis
=====================

ArXivist-generated reproduction of:

    Quantum Cryptanalysis on IBM Quantum Hardware: Extending Even-Mansour
    Period Recovery from N=4 to N=10
    Kim, Hong, Kim, Choi, Jang, Shin, Kim (arXiv:2607.18340)

IMPORTANT: the paper's central real-hardware noise-scaling technique
(enabling clean-ish Even-Mansour recovery at N=6-10) is explicitly withheld
pending an IP decision. This repo reproduces the DISCLOSED algorithms only
(Bernstein-Vazirani, Grover, and Simon's algorithm applied to Even-Mansour,
CBC-MAC forgery, and 3-round Feistel) and defaults to a noiseless simulator,
where these textbook algorithms succeed by construction. See README.md.
"""

__version__ = "0.1.0"
