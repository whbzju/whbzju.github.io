---
title: "浅析JNI（二）"
date: 2013-06-30T09:55:00+08:00
tags: ["JNI"] 
draft: false
toc: true
---

上文：[浅析JNI](http://whbzju.github.io/blog/2013/06/26/qianxi-jni/)中提到，静态注册方法有不少弊端，和现在的链接方式方式分静态链接和动态链接相识，jni技术中还有动态注册，本文将详细介绍其实现机制和原理。

#动态注册

##Java层
先看代码，为hello-jni的java层添加动态注册的native方法，与静态注册的native方法比较。

public class jniActivity extends Activity {

    private TextView tv;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setContentView(R.layout.main);

        TextView  tv = new TextView(this);
        //tv.setText( stringFromJNI() );
        tv.setText(DynamicStringFromJNI());
        setContentView(tv);
    }

    // 这个方法采用静态注册，参考ndk自带的例子hello-jni的实现
    public native String  stringFromJNI();

    // 修改成动态注册
    public native String DynamicStringFromJNI();

    // 提供方法，让native层调用
    public void testMethodForNativeCallJava(){
        Toast.makeText(getApplicationContext(), "Call from Native", Toast.LENGTH_SHORT).show();
    }

    // 加载jni库
    static{
       System.loadLibrary("learn-jni");
    }
}

以上的代码很简单，stringFromJNI上上节中用来演示静态注册，testMethodForNativeCallJava是后面要用到，暂时不用关心。我们来看`public native String DynamicStringFromJNI()`。动态注册和静态注册对java层来说没有区别。

##C层
同样，先看代码：


#include <string.h>
#include <jni.h>
#include <assert.h>
#include <android/log.h>


#define TAG "Learn-jni"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__))

/* This is a trivial JNI example where we use a native method
 * to return a new VM String. See the corresponding Java source
 * file located at:
 *
 *   apps/samples/hello-jni/project/src/com/example/hellojni/HelloJni.java
 */

jstring
Java_com_example_learn_jni_jniActivity_stringFromJNI( JNIEnv* env,
                                                  jobject thiz )
{
    return (*env)->NewStringUTF(env, "Hello from JNI !");
}


jstring dynamicHello(JNIEnv *env, jobject thiz){

    return (*env)->NewStringUTF(env, "Hello from Dynamic register");
}

static JNINativeMethod gMethods[] = {
    {
    "DynamicStringFromJNI",
    "()Ljava/lang/String;",
    (void *)dynamicHello
    },

};

int registerNativeMethods(JNIEnv* env){
    const char* className = "com/example/learn_jni/jniActivity";
    if(jniRegisterNativeMethods(env, className, gMethods, sizeof(gMethods) / sizeof(gMethods[0]))){
        return JNI_FALSE;
    }

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","jniRegisterReturn 0, registerNative return 1");
    return JNI_TRUE;
}

int jniRegisterNativeMethods(JNIEnv* env, const char* className, const JNINativeMethod* gMethods, int numMethods){
    jclass clazz;
    clazz = (*env)->FindClass(env, className);

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","call jniRegisterNative");
    if(clazz == NULL){
        //printf("clazz is null\n");

        __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","clazz is null");
        return JNI_FALSE;
    }

    if((*env)->RegisterNatives(env, clazz, gMethods, numMethods) < 0){
        __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","RegisterNative failed");
        return -1;
    }

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","jniRegister Return 0");
    return 0;
}

jint JNI_OnLoad(JavaVM* vm, void* reserved){

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","call java jni_onload");
    //printf("JNI OnLoad");
    assert(vm != NULL);

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","vm ok");
    JNIEnv* env = NULL;
    jint result = -1;

    //printf("JNI ONLoad start\n");
    //printf("JNI ONLoad start\n");

    if((*vm)->GetEnv(vm, (void**) &env, JNI_VERSION_1_4) != JNI_OK){

        __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","jni version failed");
        return -1;
    }

    //printf("JNI ONLoad\n");

    assert(env != NULL);

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","env ok");

    if(!registerNativeMethods(env)){
        __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","register failed");
        return -1;
    }

    return JNI_VERSION_1_4;
}


代码有点长，其中类似`__android_log_print(ANDROID_LOG_DEBUG,"hello-jni","env ok");`是我用来调试用的，jni方法无法直接用ide调试，一般情况下可以用打log的方式调试，如上面的做法，还需要修改Android.mk，加入`LOCAL_LDLIBS :=-llog`，放在`include $(CLEAR_VARS)
`后面。不过打log的方式有时候很低效，有另外一种调试方法是通过ndk-gdb来调试native的函数，当然这又是另外一个话题，针对它我还有一些疑问，哪天解决了再写一篇博客吧。

###JNI签名介绍
不要被上面的代码吓到，我们来理一理思路。既然是动态注册，那是不是应该有类似符号表的东西来保存java层和native层之间的对应关系呢？答案是肯定的，它叫JNINativeMethod，其定义如下：
	typedef struct{
		const char* name;
		const char* signature;
		void*       fnPtr;
	} JNINativeMethod;

30~37行的代码即做了这个事情，这个逻辑上很好理解，唯一奇怪的是里面的内容。这里和签名的格式有关，网上类似的介绍很多，简单来讲，JNI类型签名定义如下：
	（参数1类型标示;参数2类型标示;…参数n类型标示）返回值类型标示
其对应的类型标示可以参考jni.h中的定义，理解起来就是将java中的类型翻译成c中的类型：

# include <inttypes.h>      /* C99 */
typedef uint8_t         jboolean;       /* unsigned 8 bits */
typedef int8_t          jbyte;          /* signed 8 bits */
typedef uint16_t        jchar;          /* unsigned 16 bits */
typedef int16_t         jshort;         /* signed 16 bits */
typedef int32_t         jint;           /* signed 32 bits */
typedef int64_t         jlong;          /* signed 64 bits */
typedef float           jfloat;         /* 32-bit IEEE 754 */
typedef double          jdouble;        /* 64-bit IEEE 754 */
#else
typedef unsigned char   jboolean;       /* unsigned 8 bits */
typedef signed char     jbyte;          /* signed 8 bits */
typedef unsigned short  jchar;          /* unsigned 16 bits */
typedef short           jshort;         /* signed 16 bits */
typedef int             jint;           /* signed 32 bits */
typedef long long       jlong;          /* signed 64 bits */
typedef float           jfloat;         /* 32-bit IEEE 754 */
typedef double          jdouble;        /* 64-bit IEEE 754 */
#endif

更详细的对应关系请参考：<http://blog.csdn.net/lizhiguo0532/article/details/7219357>

###注册
这里我实现了两个方法：`registerNativeMethods`和`jniRegisterNativeMethods`, 最终调用`RegisterNatives`完成注册。看起来好像很复杂，其实核心只有两个函数，总结如下：

* jclass clazz = (*env)->FindClass(env, className);
* (*env)->RegisterNatives(env, clazz, gMethods, numMethods)

***PS:这里有个env，它是指向JNIEnv的一个结构体，它非常重要，后面会有针对它的讨论。***

那么动态注册的函数在什么时候和什么地方被调用，引用《深入理解Android》卷一中的原文：
> 当Java层通过System.loadLibrary加载完JNI动态库后，紧接着会查找该库中一个叫JNI_Load的函数，如果有，就调用它，而动态注册的工作是在这里完成的。

对照上面的代码中JNI_Load的实现，来理解这句话。

到这里动态注册就完成了。结果如下
![运行截图](images/androidscreen1.png)

#未完待续
