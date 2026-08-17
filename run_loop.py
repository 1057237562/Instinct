import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules["model.model_instinct"] = importlib.import_module("model.model_instinct_loop")
target = os.path.abspath(sys.argv.pop(1))
os.chdir(os.path.dirname(target))
__file__ = target
exec(open(target).read())
