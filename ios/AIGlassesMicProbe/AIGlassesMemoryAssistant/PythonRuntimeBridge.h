#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface PythonRuntimeBridge : NSObject
+ (nullable NSString *)startWithPythonHome:(NSURL *)pythonHome
                                moduleRoot:(NSURL *)moduleRoot
                                configJSON:(NSString *)configJSON
                                     error:(NSError **)error;
+ (nullable NSString *)callFunction:(NSString *)function
                          arguments:(NSArray<NSString *> *)arguments
                              error:(NSError **)error;
+ (void)stop;
@end

NS_ASSUME_NONNULL_END
