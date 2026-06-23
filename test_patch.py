import sys
import types
module = types.ModuleType("my_module")
code = """
def func():
    print("original")

def caller():
    func()
"""
exec(code, module.__dict__)
sys.modules["my_module"] = module

import my_module
my_module.func = lambda: print("patched")
my_module.caller()
