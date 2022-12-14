---
title: "浅析JNI（三）"
date: 2013-07-27T22:41:00+08:00
categories: 
draft: false
toc: true
---

没想到这个话题写到第三篇博客，写blog真费时，我的周末没了。接上文继续：
PS更新:这篇blog一拖就是一个月，主要是我太懒了。其次是在公司写的东西不能带出来，每次要写这方面的资料，必须重写一些demo进行说明，重复劳动，没有动力。

#内容概述
* JNI数据类型
* JNIEnv介绍

##JNI数据类型介绍
1. 基本数据类型
先来看一张图：
![JNI基本数据类型](/images/jni_base_type.png)

可以明显的看出JNI的数据类型只是比java的基本数据类型多了个j。


2. 引用数据类型
同样看图：
![JNI引用数据类型图](/images/jni_ref_type.png)

可以看出所有的引用类型都是jobject，和java类似。不过jni里面对jstring单独做了处理，就叫jstring类型，估计是用的频率太高，如果也是jobject会涉及到大量的“装箱拆箱“吧。

##JNIEnv介绍
JNIEnv是jni中举足轻重的一个角色，env可以理解成window中的句柄，线程中的线程描述符，或者简单理解成当前的上下文环境变量。在java VM中，它是一个局部引用，因此无法作为全局引用保存下来，每次在jni调用时，都要重新获取下env，因为env有可能会发生变化。

##实例讲解--Jni回调java
在工作中，我们使用jni时，常常会碰到需要从native回调java的需求，比如java通过jni调用native的一些函数，如果这些函数较为耗时，经常会起一个线程来完成任务，那么当任务完成时，必然要告诉java层。通常的做法是java层通过jni设置回调函数，native通过jni回调java。先看代码：

public class jniActivity extends Activity {

    private TextView tv;
    //private String test;
    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setContentView(R.layout.main);

        TextView  tv = new TextView(this);
        SetJniCallBack();
        //tv.setText( stringFromJNI() );
        tv.setText(DynamicStringFromJNI());
        setContentView(tv);
    }

    // 这个方法采用静态注册，参考ndk自带的例子hello-jni的实现
    public native String  stringFromJNI();

    // 修改成动态注册
    public native String DynamicStringFromJNI();
	
	// Java通过该方法设置回调方法
    public native void SetJniCallBack();

    // 提供方法，让native层调用
    public void testMethodForNativeCallJava(){
       Toast.makeText(getApplicationContext(), "Call from Native", Toast.LENGTH_SHORT).show();
    }

    // 加载jni库
    static{
       System.loadLibrary("learn-jni");
    }
}

C的代码和前面两篇类似，我只贴出增加的代码。

JavaVM* g_javaVm = NULL;
jobject g_object = NULL;

// jni中真正回调java的函数
void CallBack()
{
    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","callback enter");
    JNIEnv* env = NULL;
    int isAttach = 0;
    int result = (*g_javaVm)->GetEnv(g_javaVm, (void**) &env, JNI_VERSION_1_4);

    if(result != JNI_OK){
        result = (*g_javaVm)->AttachCurrentThread(g_javaVm, &env, NULL);
        if(result < 0)
            return;

        isAttach = 1;
    }

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","callback getEnv sucess");
    jclass jclazz = (*env)->GetObjectClass(env, g_object);

    if(jclazz == NULL){
        __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","callback jclazz is NULL");
        return;
    }

    jmethodID methodID = (*env)->GetMethodID(env, jclazz, "testMethodForNativeCallJava", "()V");

    if(methodID == NULL){
        __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","callback methodID is null");
        return;
    }

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","callback getMethod sucess");
    (*env)->CallVoidMethod(env, g_object, methodID);

    if(isAttach)
        (*g_javaVm)->DetachCurrentThread(g_javaVm);
    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","callback leave");
}

// 为了演示方便，在这里调用callback，现实中应该是由jni再回调native完成。
jstring dynamicHello(JNIEnv *env, jobject thiz){

    CallBack();
    return (*env)->NewStringUTF(env, "Hello from Dynamic register");
}

// java通过该函数设置回调，主要是保存一些参数，正常做法是还有一些参数，比如保存注册回调的函数名，我这里为了方便简单实现了。
void JavaSetCallBack(JNIEnv* env, jobject thiz)
{
    g_object = (*env)->NewGlobalRef(env, thiz);
    //JNIEnv *env = NULL;
}

jint JNI_OnLoad(JavaVM* vm, void* reserved){

    __android_log_print(ANDROID_LOG_DEBUG,"hello-jni","call java jni_onload");
    //printf("JNI OnLoad");
    assert(vm != NULL);

	// 保存java虚拟机
    g_javaVm = vm;
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

##分析
从上面代码中可以看到，我在OnLoad中保存了java虚拟机的指针，后续的一系列参数从该指针中获取。在JavaSetCallBack中将java对象保存下来，由于我的回调函数不是静态的，需要java的实例才能调用，所以这里需要保存java的object，又由于在java虚拟机中，jobject是一个局部引用，因此需要主动new一个全局引用作为保存。在CallBack中通过g_object重新找到类的定义，相对于c，可以理解成找到符号表。最后通过callVoidMethod调用java的方法。

##总结
JNI在实际开发中经常会碰到问题，由于其调试不方便，写起来还是蛮痛苦的。但由于其写法固定，比如函数注册等，写法很统一，常见的需求都有相应的套路，降低了开发难度。JNI部分的基础知识讲的差不多了，但是楼主还是不敢说对jni有多了解，这一个多月来，每次觉得掌握的差不多了，总又能碰到新的问题。所以只能在实际开发中不断积累经验。若后面有时间有心情，再写一篇JNI FQA吧。
