import ctypes

support = ctypes.CDLL("/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

print(support.MTLCodeGenServiceBuildRequest(0))
