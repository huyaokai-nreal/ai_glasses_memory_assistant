#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface PythonRuntimeBridge : NSObject
+ (nullable NSString *)startWithPythonHome:(NSURL *)pythonHome
                                moduleRoot:(NSURL *)moduleRoot
                                configJSON:(NSString *)configJSON
                                     error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
