
import e2b, inspect
from e2b import Sandbox
print("e2b file:", e2b.__file__)
print("Sandbox:", Sandbox)
print("has create:", hasattr(Sandbox, "create"), callable(getattr(Sandbox, "create", None)))
try:
    print("create sig:", inspect.signature(Sandbox.create))
except Exception as e:
    print("create sig err:", e)
