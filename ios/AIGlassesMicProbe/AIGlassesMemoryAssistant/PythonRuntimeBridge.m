#import "PythonRuntimeBridge.h"
#import <Python/Python.h>

@implementation PythonRuntimeBridge

+ (NSString *)startWithPythonHome:(NSURL *)pythonHome
                        moduleRoot:(NSURL *)moduleRoot
                        configJSON:(NSString *)configJSON
                             error:(NSError **)error {
    @synchronized(self) {
        if (!Py_IsInitialized()) {
            NSString *pythonPath = [NSString stringWithFormat:@"%@:%@/lib/python3.11/lib-dynload", moduleRoot.path, pythonHome.path];
            setenv("PYTHONHOME", pythonHome.fileSystemRepresentation, 1);
            setenv("PYTHONPATH", pythonPath.fileSystemRepresentation, 1);
            Py_Initialize();
        }
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject *module = PyImport_ImportModule("ai_glasses_memory_assistant.mobile_runtime");
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
