#import "PythonRuntimeBridge.h"
#import <Python/Python.h>

@implementation PythonRuntimeBridge

static NSString * const kRuntimeModule = @"ai_glasses_memory_assistant.mobile_runtime";

+ (NSString *)startWithPythonHome:(NSURL *)pythonHome
                        moduleRoot:(NSURL *)moduleRoot
                        configJSON:(NSString *)configJSON
                             error:(NSError **)error {
    @synchronized(self) {
        if (!Py_IsInitialized()) {
            NSString *pythonPath = [NSString stringWithFormat:@"%@:%@/lib/python3.11/lib-dynload", moduleRoot.path, pythonHome.path];
            setenv("PYTHONHOME", pythonHome.fileSystemRepresentation, 1);
            setenv("PYTHONPATH", pythonPath.fileSystemRepresentation, 1);
            setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
            setenv("PYTHONNOUSERSITE", "1", 1);
            Py_Initialize();
        }
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject *module = PyImport_ImportModule(kRuntimeModule.UTF8String);
        if (module == NULL) {
            PyGILState_Release(gil);
            return [self fail:error];
        }
        PyObject *function = PyObject_GetAttrString(module, "start");
        PyObject *arguments = Py_BuildValue("(s)", configJSON.UTF8String);
        PyObject *result = function == NULL ? NULL : PyObject_CallObject(function, arguments);
        Py_XDECREF(arguments);
        Py_XDECREF(function);
        Py_DECREF(module);
        if (result == NULL) {
            PyGILState_Release(gil);
            return [self fail:error];
        }
        PyObject *text = PyObject_Str(result);
        NSString *value = text == NULL ? nil : [NSString stringWithUTF8String:PyUnicode_AsUTF8(text)];
        Py_XDECREF(text);
        Py_DECREF(result);
        PyGILState_Release(gil);
        if (value == nil) {
            if (error != NULL) *error = [NSError errorWithDomain:@"AIGlassesPython" code:2 userInfo:@{NSLocalizedDescriptionKey: @"Python runtime returned invalid UTF-8"}];
            return nil;
        }
        return value;
    }
}

+ (NSString *)callFunction:(NSString *)function arguments:(NSArray<NSString *> *)arguments error:(NSError **)error {
    @synchronized(self) {
        if (!Py_IsInitialized()) {
            if (error != NULL) *error = [NSError errorWithDomain:@"AIGlassesPython" code:3 userInfo:@{NSLocalizedDescriptionKey: @"Python runtime is not running"}];
            return nil;
        }
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject *module = PyImport_ImportModule(kRuntimeModule.UTF8String);
        PyObject *callable = module == NULL ? NULL : PyObject_GetAttrString(module, function.UTF8String);
        PyObject *tuple = PyTuple_New(arguments.count);
        for (NSUInteger index = 0; index < arguments.count; index++) {
            PyTuple_SET_ITEM(tuple, index, PyUnicode_FromString(arguments[index].UTF8String));
        }
        PyObject *result = callable == NULL ? NULL : PyObject_CallObject(callable, tuple);
        Py_DECREF(tuple);
        Py_XDECREF(callable);
        Py_XDECREF(module);
        if (result == NULL) {
            NSString *failure = [self fail:error];
            PyGILState_Release(gil);
            return failure;
        }
        PyObject *text = PyObject_Str(result);
        NSString *value = text == NULL ? nil : [NSString stringWithUTF8String:PyUnicode_AsUTF8(text)];
        Py_XDECREF(text);
        Py_DECREF(result);
        PyGILState_Release(gil);
        if (value == nil && error != NULL) *error = [NSError errorWithDomain:@"AIGlassesPython" code:2 userInfo:@{NSLocalizedDescriptionKey: @"Python runtime returned invalid UTF-8"}];
        return value;
    }
}

+ (void)stop {
    @synchronized(self) {
        if (!Py_IsInitialized()) return;
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject *module = PyImport_ImportModule(kRuntimeModule.UTF8String);
        PyObject *function = module == NULL ? NULL : PyObject_GetAttrString(module, "stop");
        if (function != NULL) {
            PyObject *result = PyObject_CallObject(function, NULL);
            Py_XDECREF(result);
        }
        Py_XDECREF(function);
        Py_XDECREF(module);
        PyGILState_Release(gil);
    }
}

+ (NSString *)fail:(NSError **)error {
    PyObject *type = NULL;
    PyObject *value = NULL;
    PyObject *traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    PyObject *text = value == NULL ? NULL : PyObject_Str(value);
    NSString *message = text == NULL ? @"Python runtime failed" : [NSString stringWithUTF8String:PyUnicode_AsUTF8(text)];
    Py_XDECREF(text);
    Py_XDECREF(traceback);
    Py_XDECREF(value);
    Py_XDECREF(type);
    if (error != NULL) *error = [NSError errorWithDomain:@"AIGlassesPython" code:1 userInfo:@{NSLocalizedDescriptionKey: message ?: @"Python runtime failed"}];
    return nil;
}

@end
