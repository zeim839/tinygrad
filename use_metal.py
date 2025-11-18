from tinygrad.helpers import tqdm
import ctypes, ctypes.util

CoreServices = ctypes.CDLL(ctypes.util.find_library("CoreServices"))

def blockify(fn, rtype, *argtypes):
  class Descriptor(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
      ('reserved', ctypes.c_uint64),
      ('size', ctypes.c_uint64),
      ('copy_helper', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None))),
      ('dispose_helper', ctypes.CFUNCTYPE(None, ctypes.POINTER(None))),
      ('signature', ctypes.POINTER(ctypes.c_char)),
    ]

  class Literal(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
      ('isa', ctypes.POINTER(None)),
      ('flags', ctypes.c_int32),
      ('reserved', ctypes.c_int32),
      ('invoke', ctypes.CFUNCTYPE(rtype, ctypes.POINTER(Descriptor), *argtypes)),
      ('descriptor', ctypes.POINTER(Descriptor)),
    ]

  return ctypes.pointer(Literal(
    isa = ctypes.addressof(CoreServices._NSConcreteGlobalBlock),
    flags = (1 << 28), # BLOCK_IS_GLOBAL
    reserved = 0,
    invoke = ctypes.CFUNCTYPE(rtype, ctypes.POINTER(Descriptor), *argtypes)(lambda *args: fn(*args[1:])),
    descriptor = ctypes.pointer(Descriptor(
      reserved = 0,
      size = ctypes.sizeof(Literal),
      copy_helper = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None))(0),
      dispose_helper = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))(0),
      signature = None
    )),
  ))

#src = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e"
src = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_x86_64h"
dst = "./tmp/libraries"

dsc_extractor = ctypes.CDLL("/usr/lib/dsc_extractor.bundle")

dsc_extractor.dyld_shared_cache_extract_dylibs_progress.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
dsc_extractor.dyld_shared_cache_extract_dylibs_progress.restype = ctypes.c_int

bar = tqdm(desc='Extracting', unit=' libraries')
def progress(a, b):
  bar.t = b
  bar.update(a-bar.n+1)

dsc_extractor.dyld_shared_cache_extract_dylibs_progress(src.encode('ascii'), dst.encode('ascii'), blockify(progress, None, ctypes.c_int, ctypes.c_int))
