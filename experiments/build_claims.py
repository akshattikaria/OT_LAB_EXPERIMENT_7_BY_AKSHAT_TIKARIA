"""
Wrapper for report/build_claims.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "report")))
import build_claims

if __name__ == "__main__":
    build_claims.main()
