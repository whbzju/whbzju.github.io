export const zhihuBlogPosts = [
  {
    "slug": "weshop-ai-models",
    "titleZh": "AI 技术代替电商模特，现在可以实现了吗？",
    "titleEn": "Can AI Replace E-commerce Models?",
    "date": "2025-03-20",
    "source": "Zhihu answer",
    "sourceUrl": "https://www.zhihu.com/question/590884963/answer/2983851310",
    "excerptZh": "围绕 WeShop、AI 模特、虚拟试衣和电商商拍，讨论生成式 AI 在真实电商工作流里的可用性边界。",
    "excerptEn": "A WeShop-grounded essay on AI models, virtual try-on, e-commerce photography, and why usefulness arrives before perfect control.",
    "translationMode": "adapted",
    "blocksEn": [
      {
        "type": "p",
        "html": "This answer has changed as the product changed. When I first wrote it, the honest conclusion was: diffusion models could already produce impressive images, but they were still far from a foolproof e-commerce workflow. Since then, we have turned a lot of that exploration into WeShop, including a commercial Virtual Try-On system rather than just a demo."
      },
      {
        "type": "p",
        "html": "The current WeShop workflow can change faces, backgrounds, poses, and even rotate a garment presentation. You can try the demo on <a href=\"https://huggingface.co/spaces/WeShopAI/WeShopAI-Virtual-Try-On\">Hugging Face</a>, or use the commercial product and API through <a href=\"https://www.weshop.com\">WeShop</a>."
      },
      {
        "type": "image",
        "src": "/images/blog/weshop-ai-models-01.webp",
        "alt": "WeShop virtual try-on examples",
        "width": 1080,
        "height": 4384,
        "caption": "Examples from the WeShop virtual try-on workflow."
      },
      {
        "type": "h2",
        "html": "The important problem is not whether the image looks good once"
      },
      {
        "type": "p",
        "html": "E-commerce is not a purely digital scenario. A product photo is connected to fulfillment, consumer trust, returns, and long-term brand credibility. If an AI image changes the product detail, the image may look beautiful but still hurt the buyer experience. That is why the hardest part is not generating a good-looking model. It is preserving the actual product."
      },
      {
        "type": "p",
        "html": "This is also where many early demos became misleading. A few screenshots could make the technology look ready, but a merchant needs repeatable control: keep the garment, change the model, change the scene, keep the SKU reliable, and do it at scale."
      },
      {
        "type": "h2",
        "html": "The technical routes we tried"
      },
      {
        "type": "p",
        "html": "Midjourney was polished but too closed for this use case. Stable Diffusion WebUI was much more useful because it exposed the workflow and had an active ecosystem. DreamBooth and LoRA were helpful for injecting a specific person, product, or style into generation, but product details were still fragile. LoRA could learn that a piece of lingerie had patterns, for example, while still losing the structure of the lower band."
      },
      {
        "type": "p",
        "html": "ControlNet improved structure preservation. Inpainting made the model-replacement path more realistic because we could keep the original product image and only change the model. But inpainting introduced another product problem: masks, stages, uncertainty, and debugging cost. It was powerful for studios, but too complicated for a normal merchant."
      },
      {
        "type": "p",
        "html": "We also looked at the image-editing literature: ControlNet, Prompt-to-Prompt, Null-text Inversion, Pix2Pix-Zero, InstructPix2Pix, SDEdit, Composer, and related work. Some of these directions were promising, but e-commerce has its own instruction distribution. A model trained on general editing instructions does not automatically understand what a merchant means by “keep the collar”, “do not change the strap”, or “make this red dress green without changing the cut”."
      },
      {
        "type": "h2",
        "html": "Productization means narrowing freedom"
      },
      {
        "type": "p",
        "html": "A raw generation interface gives users freedom, but it also gives them too many ways to fail. For WeShop, we started to design templates: common scenes, parameter sets, and workflows that compress the best practices we discovered through repeated testing. This sacrifices some creative freedom, but it makes the result more controllable."
      },
      {
        "type": "p",
        "html": "That is the pattern I keep seeing in AI applications: early technology expands the possibility space, while product work narrows that space into something repeatable, legible, and safe enough for real users."
      },
      {
        "type": "h2",
        "html": "The business window"
      },
      {
        "type": "p",
        "html": "Even an imperfect intermediate product can move an industry forward. Cross-border sellers, apparel manufacturers, wholesalers, and small brands all face real production costs. If AI can reduce part of the photo-shooting cost while preserving enough trust, it is already useful."
      },
      {
        "type": "p",
        "html": "But the long-term value will not come from one more image-generation demo. It will come from workflow depth: product detail preservation, model consistency, background control, QA, API delivery, data accumulation, and the ability to serve real merchants repeatedly."
      }
    ],
    "blocksZh": [
      {
        "type": "p",
        "html": "更新下我们已经商用的Virtual TryOn 技术（不是DEMO!），换脸 换背景 换姿势都可以，转个身也可以。"
      },
      {
        "type": "p",
        "html": "想玩一下可以去Huggingface：<a href=\"https://huggingface.co/spaces/WeShopAI/WeShopAI-Virtual-Try-On\">https://huggingface.co/spaces/WeShopAI/WeShopAI-Virtual-Try-On</a>"
      },
      {
        "type": "p",
        "html": "想商用可以直接去官网(提供API服务)：<a href=\"https://www.weshop.com\">https://www.weshop.com</a>"
      },
      {
        "type": "p",
        "html": "详细介绍：<a href=\"https://www.weshop.com/blog/post-10661\">https://www.weshop.com/blog/post-10661</a>"
      },
      {
        "type": "p",
        "html": "PS：预计未来一两周内，我们会更新商品细节保持更好的版本<br>更新于2025.3.20"
      },
      {
        "type": "image",
        "src": "/images/blog/weshop-ai-models-01.webp",
        "alt": "WeShop 虚拟试衣效果示例",
        "width": 1080,
        "height": 4384,
        "caption": "WeShop 虚拟试衣与 AI 模特效果示例。"
      },
      {
        "type": "p",
        "html": "在这篇回答后，我们做了个产品，肝了一个月终于有点成绩和大家见面，效果比上个月好多了，老规矩先看下效果，目前产品开放测试：www.weshop.com"
      },
      {
        "type": "p",
        "html": "几种常见需求场景梳理："
      },
      {
        "type": "p",
        "html": "假人台转真人模特，适合服装制造商、批发商等需要给大量SKU拍照的场景。<br>真人实拍换模特、换背景，适合普通服装品牌、出海卖家等。<br>实物商品添加背景、场景等，适合大部分零售商家。<br>借助人台"
      },
      {
        "type": "p",
        "html": "高级复杂服装"
      },
      {
        "type": "p",
        "html": "高难度人物姿势"
      },
      {
        "type": "p",
        "html": "高难度姿势"
      },
      {
        "type": "p",
        "html": "多人"
      },
      {
        "type": "p",
        "html": "多人"
      },
      {
        "type": "p",
        "html": "简单的绿幕背景"
      },
      {
        "type": "p",
        "html": "非全身人台"
      },
      {
        "type": "p",
        "html": "已有商品图片换模特"
      },
      {
        "type": "p",
        "html": "换背景换表情"
      },
      {
        "type": "p",
        "html": "换年龄"
      },
      {
        "type": "p",
        "html": "换模特人种肤色"
      },
      {
        "type": "p",
        "html": "大码模特也没有问题"
      },
      {
        "type": "p",
        "html": "换金发美女"
      },
      {
        "type": "p",
        "html": "换亚裔，亚洲人目前还有不少场景需要优化"
      },
      {
        "type": "p",
        "html": "商品换背景"
      },
      {
        "type": "p",
        "html": "本身有背景商品换背景"
      },
      {
        "type": "p",
        "html": "白底商品图（PNG）换背景"
      },
      {
        "type": "p",
        "html": "以上是我们近期在不断实践新技术的结果展示，当然已经有一批内测客户体验过我们的产品。目前AI还处于早期的阶段，不同的使用姿势，效果差异很大，我们团队一直在不断迭代，尝试用产品化的功能去沉淀最佳实践，降低客户使用AI产品的门槛。同时我们整个团队也不断被AI的能力边界所震撼，依旧在快速的成长中，欢迎对我们产品感兴趣的朋友们去官网加小助手和我们交流："
      },
      {
        "type": "p",
        "html": "-----------------------------------5月13号更新----------------------------------------------"
      },
      {
        "type": "p",
        "html": "先看一个场景，国内的商家在出海时，常常要面对重新请他国的模特重新拍摄商品照片的问题，如果能一键变换不同国家的模特而保持商品不变，则能降低不少营销侧的成本。给大家看下这两周和几个小伙伴一起搞的一个demo效果："
      },
      {
        "type": "p",
        "html": "说下结论，diffusion models虽然已经能生成出非常惊艳的效果，但其在精准和控制上依旧离傻瓜式的产品体验有明显的距离。将技术产品化的过程中，不仅仅是基础模型的创新，也存在工程、场景适配的调参、不同模型的融合等大量的具体工作，需要更多相关的从业者投入其中。就算是不成熟的中间态产品，也能对行业起到不错的推动作用。"
      },
      {
        "type": "h2",
        "html": "背景"
      },
      {
        "type": "p",
        "html": "随着AI技术的持续出圈，电商圈的小伙伴也很积极在尝试各种可能性，估计很多从业者在各种社交媒体刷到过下面一些图："
      },
      {
        "type": "h2",
        "html": "基于Diffusion技术有明显的特点："
      },
      {
        "type": "p",
        "html": "生成效果更加逼真，具备接近真实图片的观感，<br>通过自然语言来描述需求，即常说的prompt，自由度很高。"
      },
      {
        "type": "p",
        "html": "但如果大家仔细看上面的图片，也很容易发现问题，商品图片的细节被改变了。在今天的电商业务中，拍摄成本确实一个明显的成本项，若有新的技术能够帮助大家优化其中成本，体现在消费者侧则是可进一步降低售价。但电商它不是一个纯数字化场景，最终需要实物履约，消费者的购物体验经常被货不对板伤害，比如有些商家过度P图，更甚的是有些商家直接无货空挂，靠图片测款，有了订单再想办法找补货。因此，若新技术的产品化程度不高，则一定会伤害到用户体验，如果靠牺牲用户体验来达成该成本的优化，从长期看并不是一个有意义的事情。"
      },
      {
        "type": "h2",
        "html": "快速梳理下现有技术的方案"
      },
      {
        "type": "p",
        "html": "考虑到今天AI技术的进步是以天为单位在更新，现在有缺陷的技术不代表未来不能解决。梳理一波现有技术的方案，有助于我们理解如何开展下一步的创新，但并不是说这个技术路线的正确性。"
      },
      {
        "type": "p",
        "html": "工具选型：MidJourney vs Stable diffusion webui<br>依靠MidJourney，MidJourney的产品化程度很高，导致自由度也比较低。一般是用它的img2img来做，经常需要用PS做一些mask图，整体效果不太可控，个人认为可行性最差。<br>基于开源的stable diffusion webui项目，该项目是在stable diffusion社区基础上做的一个集成工作，应该是目前最流行、各项feature集成度最高、社区最活跃的项目，在github上已经有63k的star。"
      },
      {
        "type": "p",
        "html": "目前已经有很多产品都是基于webui做二次开发。"
      },
      {
        "type": "h2",
        "html": "文本驱动生成：Dreambooth + LoRA的方式"
      },
      {
        "type": "p",
        "html": "diffusion难点是准确控制生成想要的特定物体，Google提出了dreambooth的来解决这个问题。训练特色的模特或商品LoRA模型，依靠webui的feature，在text2img或者img2img时候在prompt里面插入自有的LoRA模型，从而保持一定的商品或模特的独特性。Civitai上有非常多社区贡献的LoRA模型，大家可以去感受一波。"
      },
      {
        "type": "p",
        "html": "dreambooth+lora确实能保持不少独特性，而且训练也很简单，只要10张左右的图片效果就挺好的，加上用lora的训练方式，对算力要求也不高。其效果就如论文原作者给的示例，能把作者原图中狗的样子变成一个概念注入到一个特殊的[V]中，从而可以在未来生成过程中用[V]来触发。"
      },
      {
        "type": "p",
        "html": "当然还有text inversion等方案，不过text inverison没有它方便，大家用dreambooth比较多。本质上是自然语言和图像之间存在多对多的问题，用自然语言精准的描述一个图片的所有细节是不现实的，这也是目前很多多模态模型在各个领域应用中经常会碰到的问题。"
      },
      {
        "type": "p",
        "html": "但如果大家仔细去看网上分享的case，会发现人的lora模型效果要比商品好很多，比如一些明星、二次元妹子的LoRA，反过来在商品维度，很多细节、色彩还是会有问题。在我们早期的LoRA实践中，输入的原商品如下："
      },
      {
        "type": "p",
        "html": "可以看到LoRA是学到了这家内衣会有花纹，但是下围没有了，当然在后面不停的prompt工程和调参中，也能有一些出图是有完整商品结构的，但是别的细节又会有问题，比如下面的case，左边是用于训练LoRA的商品图，右边是生成的图片。"
      },
      {
        "type": "h2",
        "html": "LoRA + ControlNet"
      },
      {
        "type": "p",
        "html": "很自然，大家就会想要ControlNet来帮忙，比如用它的Canny去做商品细节的复原，如下面两张图所示，虽然还有明显的问题，但商品的结构、花纹细节已经好很多了："
      },
      {
        "type": "h2",
        "html": "局部编辑：impaint + LoRA + ControlNet"
      },
      {
        "type": "p",
        "html": "对国内的商家来讲，请不同国家的模特拍摄成本不低，如果我们换个思路，只对已有的商品图片换模特，则有可能利用生成式模型逼真的特点同时又保留了商品的细节。下图是我们快速实践的效果："
      },
      {
        "type": "p",
        "html": "生成式模型对比过去的换脸和换肤色技术，在感官上明显更逼真，五官会更接近不同国家民族的特色。但是impaint有个致命的问题，需要去手动做mask，我们调研了不少skin detect，包括最近的segment anything、Grounding_DINO等技术，各种corner case比较多，目前还无法直接产品化。"
      },
      {
        "type": "p",
        "html": "而且从用户体验的角度，mask、impaint、img2img，stage比较多，需要用户理解的成本变大，且每个stage的生成即需要不少时间又有一定的不确定性，调试成本很高，用户体验不可控，因此，这个方案还是只能工作室玩，无法有效的产品化。"
      },
      {
        "type": "p",
        "html": "模型层面的Image Editing的相关工作"
      },
      {
        "type": "p",
        "html": "作为一个算法背景的工程师，在快速实践了网上已有的技术方案后，直觉上判断学术界肯定有很多相关的工作。我们把相关的paper过了一遍，其中比较重要的工作：ControlNet、Prompt2Prompt、Null-text Inversion、pix2pix-zero、InstructPix2Pix、SDEdit、Composer等等。其中我个人认为比较有潜力的工作是prompt2prompt和instructpix2pix，可能比较有机会在更大的数据集和算力上进一步进化。"
      },
      {
        "type": "p",
        "html": "instructpix2pix结合了prompt2prompt的想法，提出用gpt3来构造不同的prompt的edit instruction，再通过sd模型来构造出这个edit instruction的图片对，从而无中生有的构造出了大量的带对比的样本对。接着finetune了stable diffusion的model，从而让模型更容易去对齐用户的instruction。考虑到该论文的作者是学校的背景，受限于与资金和算力，只能在一个比较小的数据集上finetune，希望未来有实力更强大的团队能把它顺利scaling。"
      },
      {
        "type": "p",
        "html": "然而但由于电商场景的特殊性，在使用中的场景和isntruction构造的训练数据集存在天然差异，直接按paper里面说的姿势使用效果一般。我们也在尝试去构建电商侧的instruction数据集，finetune一个更适合电商图片编辑的model。"
      },
      {
        "type": "p",
        "html": "在实践中，我一开始选型了diffusers，对比webui的项目，它干净的多，而且是我比较熟悉的huggingface团队的工作，只需要按需求开发个新的pipeline就好了。"
      },
      {
        "type": "p",
        "html": "但是团队的另一个设计师同学，主要用webui做调参，导致我们两边调参匹配不方便。因此后面还是切到了基于webui的api做二次开发，它的api文档比较落后，直接看代码更容易理解使用姿势。"
      },
      {
        "type": "p",
        "html": "PS：diffusers的instructpix2pix的example有些问题，不过社区反应很快，我们给了issue和改进意见后，基本都是当天就修复。"
      },
      {
        "type": "p",
        "html": "我们实践过程中发现生成的效果和输入图片本身、想要的效果、模型、参数、prompt都有关系，对普通用户太不友好了。因此从用户体验出发，我们做点了产品流程的设计，预先设计一批不同参数的模板，用户可以根据需求选择合适的模板，一键生成需要的图片。这么做一定程度上损失了不少自由度，但效果的可控性会好很多。以下是我们一些模板的示例："
      },
      {
        "type": "p",
        "html": "下面是我们Demo实际run的一些case："
      },
      {
        "type": "h2",
        "html": "最后"
      },
      {
        "type": "p",
        "html": "这个项目对我个人来讲有点像AI hackathon，整个项目就两三个人，搞了2周左右，迭代速度非常快，有点10年前移动互联网刚起来时写代码的感觉。原计划开放一些内测的接口给大家测试，但我们这个项目的算力也是别人支持的，目前想出一组效果不错的图，大概需要2-4分钟左右，用户体验也不好。若未来我们能更好的解决这些体验的问题，应该会和大家见上面。"
      },
      {
        "type": "p",
        "html": "还有许多未尽的想法，也欢迎大家有想法和我们交流，如果合适，我们可以提供一些算力支持。"
      },
      {
        "type": "p",
        "html": "如何融合ControlNet与InstructPix2Pix的各自优点，一些衣服的纹理细节、一些场景的深度信息，都需要controlnet来帮忙<br>finetune出一个能够准确理解电商场景需求的diffusion model。电商垂直领域的图像文本对齐工作。包括两部分，一个是文本对齐电商侧的概念，一个是需要对stable diffusion的预训练model做finetune。"
      },
      {
        "type": "p",
        "html": "如果有同学针对上述问题有想法，请联系我们wujia@mogu.com，我们可以一起探讨下，如果合适我们愿意提供一些算力支持。另外，如果有同学对电商侧的数据感兴趣，商业合作和一些偏公益的用途，都可以联系我申请。"
      },
      {
        "type": "p",
        "html": "去年花了大半年的时间，做了一个轻量的多模态模型，它能够对大部分的电商网站做结构化的信息抽取，我们把它用在weshop这个项目中，"
      },
      {
        "type": "p",
        "html": "我们已经在全球收录了接近10亿左右的电商数据，300w左右的独立站点，其中有一半左右是非标准化的站点。WeShop项目目前还是beta状态，产品体验问题较多，们计划建立一个全网最全的电商数据库，欢迎大家给我们提意见。"
      },
      {
        "type": "p",
        "html": "若有同学对diffusion不熟悉，可以先参考我这篇导读："
      },
      {
        "type": "p",
        "html": "PS：当然项目还有很多考虑不周的情况，请多多包涵，上诉图片如有侵权，请联系我删除。"
      }
    ]
  },
  {
    "slug": "rl-recommender-systems",
    "titleZh": "增强学习在推荐系统有什么最新进展？",
    "titleEn": "What YouTube’s RL Papers Teach Recommender-System Builders",
    "date": "2019",
    "source": "Zhihu answer",
    "sourceUrl": "https://www.zhihu.com/question/57388498/answer/801574687",
    "excerptZh": "用 YouTube 推荐系统里的强化学习论文，讨论长期收益、离线策略校正、slate 推荐和推荐系统工程约束。",
    "excerptEn": "A recommender-systems reading of YouTube’s RL work: long-term reward, off-policy correction, slate recommendation, and engineering constraints.",
    "translationMode": "adapted",
    "blocksEn": [
      {
        "type": "p",
        "html": "Around that time, people in the industry started circulating the claim that YouTube had successfully applied reinforcement learning to recommendation and had achieved one of its most significant online gains in years. Two papers were especially worth reading: <em>Top-K Off-Policy Correction for a REINFORCE Recommender System</em> and <em>Reinforcement Learning for Slate-based Recommender Systems</em>."
      },
      {
        "type": "image",
        "src": "/images/blog/rl-recommender-01.png",
        "alt": "YouTube reinforcement learning recommender paper diagram",
        "width": 1032,
        "height": 350,
        "caption": "One of the paper diagrams discussed in the original Zhihu answer."
      },
      {
        "type": "h2",
        "html": "Why recommendation is tempting for RL"
      },
      {
        "type": "p",
        "html": "A recommendation system is not just trying to predict the next click. It is shaping a sequence of user experiences. A short-term click may reduce long-term satisfaction; repeated similar content may raise immediate engagement while damaging exploration; and the value of one item often depends on what else appears around it. These are exactly the kinds of problems that make reinforcement learning attractive."
      },
      {
        "type": "p",
        "html": "But recommendation is also a hard RL environment. The action space is enormous, the reward is delayed and noisy, online exploration is expensive, and most teams only have logs generated by an old behavior policy. You cannot simply take a textbook RL algorithm and plug it into a production recommender."
      },
      {
        "type": "h2",
        "html": "The off-policy problem"
      },
      {
        "type": "p",
        "html": "YouTube’s Top-K off-policy correction work starts from a practical constraint: the training data was collected by an existing system, not by the policy you now want to learn. If you optimize REINFORCE directly on those logs, you will bias the model toward what the old system chose to expose. The paper’s core contribution is a correction method that makes this logged data more usable for learning a new ranking policy."
      },
      {
        "type": "p",
        "html": "For a production team, the lesson is straightforward: before discussing RL, first ask how the logs were generated, what the exposure policy was, and whether your reward can be trusted. Most “RL for recommendation” failures begin before the model is trained."
      },
      {
        "type": "h2",
        "html": "The slate problem"
      },
      {
        "type": "p",
        "html": "The second paper looks at slate recommendation. In a real product, the user does not see one isolated item; the user sees a list. The value of the slate is not the sum of independent item values. Items interact with one another through position, substitution, diversity, and user attention."
      },
      {
        "type": "p",
        "html": "The paper proposes a tractable way to reason about slate-level reward without enumerating the impossible number of all possible slates. That is important because the combinatorial action space is one of the main reasons recommender-system RL looks elegant in theory and painful in practice."
      },
      {
        "type": "h2",
        "html": "What I take from it"
      },
      {
        "type": "p",
        "html": "The practical path is not to worship the word “RL”. It is to identify where a recommender is being hurt by short-term objectives, where sequence effects matter, and where logged feedback can support a safer learning loop. Sometimes that means reinforcement learning; sometimes it means bandits, better counterfactual evaluation, better reward design, or simply a more honest offline experiment."
      },
      {
        "type": "p",
        "html": "For most teams, the hard work is still engineering and product definition: define the reward, log exposure correctly, build offline evaluation, control exploration risk, and make sure the model can actually run inside the serving system."
      }
    ],
    "blocksZh": [
      {
        "type": "p",
        "html": "前阵子正好写了一篇专栏分析google在youtube应用强化的两篇论文："
      },
      {
        "type": "p",
        "html": "吴海波：以youtube的RL论文学习如何在推荐场景应用RL<br>284 赞同 · 14 评论 文章"
      },
      {
        "type": "p",
        "html": "正文如下："
      },
      {
        "type": "p",
        "html": "2个月前，业界开始流传youtube成功将RL应用在了推荐场景，并且演讲者在视频（<a href=\"https://www.youtube.com/watch?v=HEqQ2_1XRTs\">https://www.youtube.com/watch?v=HEqQ2_1XRTs</a>）中说是youtube近几年来取得的最显著的线上收益。"
      },
      {
        "type": "p",
        "html": "放出了两篇论文：Top-K Off-Policy Correction for a REINFORCE Recommender System和Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology。本文不想做论文讲解，已经有同学做的不错了：<a href=\"http://wd1900.github.io/2019/06/23/Top-K-Off-Policy-Correction-for-a-REINFORCE-Recommender-System-on-Youtube/\">http://wd1900.github.io/2019/06/23/Top-K-Off-Policy-Correction-for-a-REINFORCE-Recommender-System-on-Youtube/</a>。"
      },
      {
        "type": "image",
        "src": "/images/blog/rl-recommender-01.png",
        "alt": "YouTube 强化学习推荐论文示意图",
        "width": 1032,
        "height": 350,
        "caption": "原回答中讨论的 YouTube 强化学习推荐论文示意图。"
      },
      {
        "type": "p",
        "html": "个人建议两篇论文都仔细读读，TopK的篇幅较短，重点突出，比较容易理解，但细节上SlateQ这篇更多，对比着看更容易理解。而且，特别有意思的是，这两篇论文都说有效果，但是用的方法却不同，一个是off-policy，一个是value-base，用on-policy。很像大公司要做，把主流的几种路线让不同的组都做一遍，谁效果好谁上。个人更喜欢第二篇一些，会有更多的公式细节和工程实践的方案。"
      },
      {
        "type": "p",
        "html": "很多做个性化推荐的同学，并没有很多强化学习的背景，而RL又是一门体系繁杂的学科，和推荐中常用的supervised learning有一些区别，入门相对会困难一些。本文将尝试根据这两篇有工业界背景的论文，来解答下RL在推荐场景解决什么问题，又会遇到什么困难，我们入门需要学习一些哪些相关的知识点。本文针对有一定机器学习背景，但对RL领域并不熟悉的童鞋。"
      },
      {
        "type": "p",
        "html": "本文的重点如下："
      },
      {
        "type": "p",
        "html": "目前推荐的问题是什么<br>RL在推荐场景的挑战及解决方案<br>常见的套路是哪些<br>推荐系统目前的问题"
      },
      {
        "type": "p",
        "html": "目前主流的个性化推荐技术的问题，突出的大概有以下几点："
      },
      {
        "type": "p",
        "html": "优化的目标都是short term reward，比如点击率、观看时长，很难对long term reward建模。<br>最主要的是预测用户的兴趣，但模型都是基于logged feedback训练，样本和特征极度稀疏，大量的物料没有充分展示过，同时还是有大量的新物料和新用户涌入，存在大量的bias。另外，用户的兴趣变化剧烈，行为多样性，存在很多Noise。<br>pigeon-hole：在短期目标下，容易不停的给用户推荐已有的偏好。在另一面，当新用户或者无行为用户来的时候，会更倾向于用大爆款去承接。<br>RL应用在推荐的挑战"
      },
      {
        "type": "p",
        "html": "看slide"
      },
      {
        "type": "p",
        "html": "extremely large action space：many millions of items to recommend.如果要考虑真实场景是给用户看一屏的物料，则更夸张，是一个排列组合问题。<br>由于是动态环境，无法确认给用户看一个没有看过的物料，用户的反馈会是什么，所以无法有效模拟，训练难度增加。<br>本质上都要预估user latent state，但存在大量的unobersever样本和noise，预估很困难，这个问题在RL和其他场景中共存。<br>long term reward难以建模，且long/short term reward。tradeoff due to user state estimate error。<br>旅程开始"
      },
      {
        "type": "p",
        "html": "熟悉一个新领域，最有效率的做法是和熟悉的领域做结合。接下来，让我们先简单看下RL的基本知识点，然后从label、objective、optimization、evaluation来切入吧。"
      },
      {
        "type": "h2",
        "html": "RL的基本知识"
      },
      {
        "type": "h2",
        "html": "有一些基本的RL知识，我们得先了解一下，首先是场景的四元组结构："
      },
      {
        "type": "p",
        "html": "RL最大的特点是和环境的交互，是一种trial-error的过程，通常我们会用MDP来描述整个过程，结合推荐场景，四元组数学定义如下："
      },
      {
        "type": "p",
        "html": "• S: a continuous state space describing the user states;<br>• A: a discrete action space, containing items available for recommendation;<br>• P : S × A × S → R is the state transition probability;<br>• R : S × A → R is the reward function, where r(s, a) is the immediate reward obtained by performing action a at user state s;"
      },
      {
        "type": "h2",
        "html": "RL在推荐场景的Label特点"
      },
      {
        "type": "p",
        "html": "众所周知，RL是典型的需要海量数据的场景，比如著名的AlphaGo采用了左右互博的方式来弥补训练数据不足的问题。但是在推荐场景，用户和系统的交互是动态的，即无法模拟。举个例子，你不知道把一个没有推荐过的商品a给用户，用户会有什么反馈。"
      },
      {
        "type": "p",
        "html": "老生常谈Bias"
      },
      {
        "type": "p",
        "html": "好在推荐场景的样本收集成本低，量级比较大，但问题是存在较为严重的Bias。即只有被系统展示过的物料才有反馈，而且，还会有源源不断的新物料和用户加入。很多公司会采用EE的方式去解决，有些童鞋表示EE是天问，这个点不能说错，更多的是太从技术角度考虑问题了。"
      },
      {
        "type": "p",
        "html": "EE要解决是的生态问题，必然是要和业务形态结合在一起，比如知乎的内容自荐（虽然效果是呵呵的）。这个点估计我们公司是EE应用的很成功的一个了，前阵子居然在供应商口中听到了准确的EE描述，震惊于我们的业务同学平时都和他们聊什么。"
      },
      {
        "type": "p",
        "html": "off-policy vs on-policy"
      },
      {
        "type": "p",
        "html": "论文[1]则采取off-policy的方式来缓解。off-policy的特点是，使用了两个policy，一个是用户behavior的β，代表产生用户行为Trajectory：(s0,A0,s1, · · · )的策略，另一个是系统决策的π，代表系统是如何在面对用户a在状态s下选择某个action的。"
      },
      {
        "type": "p",
        "html": "RL中还有on-policy的方法，和off-policy的区别在于更新Q值的时候是沿用既定策略还是用新策略。更好的解释参考这里：<a href=\"https://www.zhihu.com/question/57159315\">https://www.zhihu.com/question/57159315</a>"
      },
      {
        "type": "p",
        "html": "importance weight"
      },
      {
        "type": "p",
        "html": "off-policy的好处是一定程度上带了exploration，但也带来了问题："
      },
      {
        "type": "p",
        "html": "In particular, the fact that we collect data with a periodicity of several hours and compute many policy parameter updates before deploying a new version of the policy in production implies that the set of trajectories we employ to estimate the policy gradient is generated by a different policy. Moreover, we learn from batched feedback collected by other recommenders as well, which follow drastically different policies. A naive policy gradient estimator is no longer unbiased as the gradient in Equation (2) requires sampling trajectories from the updated policy πθ while the trajectories we collected were drawn from a combination of historical policies β."
      },
      {
        "type": "p",
        "html": "常见的是引入importance weighting来解决。看下公式"
      },
      {
        "type": "p",
        "html": "从公式看，和标准的objective比，多了一个因子，因为这个因子是连乘和rnn的问题类似，梯度容易爆炸或消失。论文中用了一个近似解，并有人证明了是ok的。"
      },
      {
        "type": "h2",
        "html": "RL在推荐场景的Objective特点"
      },
      {
        "type": "p",
        "html": "在前文中，我们提到，现有的推荐技术，大多是在优化短期目标，比如点击率、停留时长等，用户的反馈是实时的。用户的反馈时长越长，越难优化，比如成交gmv就比ctr难。"
      },
      {
        "type": "h2",
        "html": "同时也说明，RL可能在这种场景更有优势。看下objective的形式表达："
      },
      {
        "type": "p",
        "html": "可以发现，最大的特点是前面有个累加符号。这也意味着，RL可以支持和用户多轮交互，也可以优化长期目标。这个特点，也是最吸引做个性化推荐的同学，大家想象下自已在使用一些个性化产品的时候，是不是天然就在做多轮交互。"
      },
      {
        "type": "p",
        "html": "轮到Bellman公式上场了，先看下核心思想："
      },
      {
        "type": "p",
        "html": "The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next."
      },
      {
        "type": "p",
        "html": "看下公式，注意它包含了时间，有助于理解。"
      },
      {
        "type": "p",
        "html": "在论文[2]中，有更多关于bellman在loss中推导的细节。由于论文[1]采用的policy-gradient的优化策略，我们需要得到loss的梯度："
      },
      {
        "type": "p",
        "html": "加入importance weighting和一些correction后，"
      },
      {
        "type": "p",
        "html": "更多细节可以去参考论文。"
      },
      {
        "type": "h2",
        "html": "optimization和evaluation"
      },
      {
        "type": "p",
        "html": "通常，RL可以分成两种，value-base和policy-base，虽然不是完全以optimial的角度看，但两种套路的优化方法有较大的区别。其中value-base虽然直观容易理解，但一直被质疑不能稳定的收敛。"
      },
      {
        "type": "p",
        "html": "they are known to be prone to instability with function approximation。"
      },
      {
        "type": "p",
        "html": "而policy-base则有较好的收敛性质，所以在很多推荐场景的RL应用，大部分会选择policy-base。当然现在也很有很多二者融合的策略，比如A3C、DDPG这种，也是比较流行的。"
      },
      {
        "type": "p",
        "html": "怎么训练β和π"
      },
      {
        "type": "p",
        "html": "π的训练是比较常规的，有意思的是β的学习。用户的behavior是很难建模的，我们还是用nn的方式去学一个出来，这里有一个单独的分支去预估β，和π是一个网络，但是它的梯度不回传，如下图："
      },
      {
        "type": "p",
        "html": "这样就不会干扰π。二者的区别如下："
      },
      {
        "type": "p",
        "html": "(1) While the main policy πθ is effectively trained using a weighted softmax to take into account of long term reward, the behavior policy head βθ′ is trained using only the state-action pairs;<br>(2) While the main policy head πθ is trained using only items on the trajectory with non-zero reward 3, the behavior policy βθ′ is trained using all of the items on the trajectory to avoid introducing bias in the β estimate."
      },
      {
        "type": "p",
        "html": "为何要把evaluation拿出来讲呢？通常，我们线下看AUC，线上直接看abtest的效果。本来我比较期待论文中关于长期目标的设计，不过论文[1]作者的方式还是比较简单，可借鉴的不多："
      },
      {
        "type": "p",
        "html": "The immediate reward r is designed to reflect different user activities; videos that are recommended but not clicked receive zero reward. The long term reward R is aggregated over a time horizon of 4–10 hours."
      },
      {
        "type": "p",
        "html": "论文[2]中没有细讲。"
      },
      {
        "type": "p",
        "html": "两篇论文中还有很大的篇幅来讲Simulation下的结果，[1]的目的是为了证明作者提出的correction和topK的作用，做解释性分析挺好的，[2]做了下算法对比，并且验证了对user choice model鲁棒，但我觉得对实践帮助不大。"
      },
      {
        "type": "h2",
        "html": "One more thing：TopK在解决什么问题？<br>listwise的问题"
      },
      {
        "type": "p",
        "html": "主流的个性化推荐应用，都是一次性给用户看一屏的物料，即给出的是一个列表。而目前主流的个性化技术，以ctr预估为例，主要集中在预估单个物料的ctr，和真实场景有一定的gap。当然，了解过learning to rank的同学，早就听过pointwise、pairwise、listwise，其中listwise就是在解决这个问题。"
      },
      {
        "type": "p",
        "html": "通常，listwise的loss并不容易优化，复杂度较高。据我所知，真正在实践中应用是不多的。RL在推荐场景，也会遇到相同的问题。但直接做list推荐是不现实的，假设我们一次推荐K个物料，总共有N个物料，那么我们能选择的action就是一个排列组合问题，C_N_K * K!个，当N是百万级时，量级非常夸张。"
      },
      {
        "type": "p",
        "html": "这种情况下，如果不做些假设，问题基本就没有可能在现实中求解了。"
      },
      {
        "type": "p",
        "html": "youtube的两篇论文，都将问题从listwise（他们叫slatewise）转化成了itemwise。但这个itemwise和我们常规理解的pointwise的个性化技术还是有区别的。在于这个wise是reward上的表达，同时要引申出user choice model。"
      },
      {
        "type": "p",
        "html": "user choice model"
      },
      {
        "type": "p",
        "html": "pointwise的方法只考虑单个item的概率，论文中提出的itemwise，虽然也是认为最后的reward只和每个被选中的item有关，且item直接不互相影响，但它有对user choice做假设。比如论文[2]还做了更详细的假设，将目标函数的优化变成一个多项式内可解的问题："
      },
      {
        "type": "p",
        "html": "这两个假设也蛮合理的，SC是指用户一次指选择一个item，RTDS是指reward只和当前选择的item有关。"
      },
      {
        "type": "p",
        "html": "有不少研究是专门针对user choice model的，一般在经济学中比较多。推荐中常见的有cascade model和mutilnomial logit model，比如cascade model，会认为用户选择某个item的概率是p，那么在一个list下滑的过程中，点击了第j个item的概率是(1-p(i))^j * p(j)."
      },
      {
        "type": "p",
        "html": "论文[1]中最后的objective中有一个因子，表达了user choice的假设："
      },
      {
        "type": "p",
        "html": "简单理解就是，用π当做用户每次选择的概率，那上面就是K-1不选择a概率的连乘。而论文[2]中，RL模型和现有的监督模型是融合在一起的，直接用pCTR模型预估的pctr来当这个user choice的概率。"
      },
      {
        "type": "h2",
        "html": "最后"
      },
      {
        "type": "p",
        "html": "这篇写的有点长，但就算如此，看了本文也很难让大家一下子就熟悉了RL，希望能起到抛砖引玉的作用吧。从实践角度讲，比较可惜的是long term reward的建模、tensorflow在训练大规模RL应用时的问题讲的很少。最后，不知道youtube有没有在mutil-task上深入实践过，论文[2]中也提到它在long term上能做一些事情，和RL的对比是怎么样的。"
      },
      {
        "type": "p",
        "html": "参考"
      },
      {
        "type": "p",
        "html": "[1] Top-K Off-Policy Correction for a REINFORCE Recommender System"
      },
      {
        "type": "p",
        "html": "[2] Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology*"
      },
      {
        "type": "p",
        "html": "[3] <a href=\"https://slideslive.com/38917655/reinforcement-learning-in-recommender-systems-some-challenges\">https://slideslive.com/38917655/reinforcement-learning-in-recommender-systems-some-challenges</a>"
      },
      {
        "type": "p",
        "html": "[4] <a href=\"https://zhuanlan.zhihu.com/p/72669137\">https://zhuanlan.zhihu.com/p/72669137</a>"
      },
      {
        "type": "p",
        "html": "[5] 强化学习中on-policy 与off-policy有什么区别？<a href=\"https://www.zhihu.com/question/57159315\">https://www.zhihu.com/question/57159315</a>"
      },
      {
        "type": "p",
        "html": "[6] <a href=\"http://wd1900.github.io/2019/06/23/Top-K-Off-Policy-Correction-for-a-REINFORCE-Recommender-System-on-Youtube/\">http://wd1900.github.io/2019/06/23/Top-K-Off-Policy-Correction-for-a-REINFORCE-Recommender-System-on-Youtube/</a>"
      }
    ]
  },
  {
    "slug": "airbnb-real-time-personalization",
    "titleZh": "如何评价Airbnb的Real-time Personalization获得2018 kdd最佳论文？",
    "titleEn": "Reading Airbnb’s Real-Time Personalization Paper as a Recommender-Systems Practitioner",
    "date": "2018",
    "source": "Zhihu column",
    "sourceUrl": "https://zhuanlan.zhihu.com/p/49537461",
    "excerptZh": "从推荐系统实践角度解读 Airbnb KDD 2018 最佳论文，重点不是 fancy 模型，而是 embedding 语料、稀疏问题、实时特征和负反馈。",
    "excerptEn": "A practitioner’s reading of Airbnb’s KDD 2018 paper: corpus construction, sparse IDs, real-time embedding features, and negative feedback.",
    "translationMode": "adapted",
    "blocksEn": [
      {
        "type": "p",
        "html": "Airbnb’s KDD 2018 best paper was not a flashy paper, and that is exactly why I liked it. It reminded me of Google’s Wide & Deep work: practical, grounded, and close to the kind of problems real recommender-system teams face every day."
      },
      {
        "type": "p",
        "html": "In recommendation, embedding work is often described as if the model is the main story. In practice, the model is only one piece. The more important questions are: what behavior sequence becomes the corpus, what the embedding space is supposed to represent, how sparse IDs are handled, how the vectors are evaluated, and how the features enter online ranking."
      },
      {
        "type": "h2",
        "html": "What should the embedding represent?"
      },
      {
        "type": "p",
        "html": "In NLP, the embedding space is usually a semantic space induced by text. In e-commerce, news, travel, or other internet products, the “corpus” is usually user behavior. Clicks, bookings, purchases, favorites, skips, and rejections do not express the same kind of interest. The embedding space is therefore not universal; it is shaped by the behavior you choose to train on."
      },
      {
        "type": "p",
        "html": "This is why session construction matters more than trying every Word2Vec variant. If a user clicks product B after searching for dresses and then product C after searching for phones, putting B and C into the same context may teach the model the wrong thing. Airbnb’s definition of a new session after a 30-minute gap is not a universal rule, but it shows the kind of product judgment required."
      },
      {
        "type": "h2",
        "html": "Bookings as global context"
      },
      {
        "type": "p",
        "html": "A nice part of the paper is the way booked listings are used as global context. A normal Word2Vec window changes as it slides through the session. Airbnb adds the eventually booked listing as a target that should be predicted regardless of whether it appears inside the local window."
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-01.jpg",
        "alt": "Airbnb embedding objective with global context",
        "width": 982,
        "height": 516,
        "caption": "Airbnb's adaptation adds the booked listing as global context."
      },
      {
        "type": "p",
        "html": "My reading is that this is a simple way to move an unsupervised embedding objective closer to the business objective. It is not just learning what was nearby in behavior; it is also nudging the vector space toward what eventually converted."
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-02.jpg",
        "alt": "Airbnb global context objective formula",
        "width": 1002,
        "height": 152,
        "caption": "The objective term that connects local behavior with the eventual booking."
      },
      {
        "type": "h2",
        "html": "Sparse IDs are still the hard part"
      },
      {
        "type": "p",
        "html": "Word2Vec is not magic. It still needs entities to appear often enough to learn useful representations. The real gain often comes from the middle-frequency items, not the head items or the long tail. Head items already have enough data for simpler item-to-item methods; tail items remain too sparse. A useful embedding system needs a large enough middle layer of entities."
      },
      {
        "type": "p",
        "html": "Airbnb filters entities with roughly five to ten occurrences, and also uses clustering-like methods for sparse IDs. The specific rule is tied to Airbnb’s business, but the principle generalizes: if ID sparsity is not handled, the rest of the modeling discussion is premature."
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-03.jpg",
        "alt": "Hashed feature column collision example",
        "width": 1224,
        "height": 470,
        "caption": "A hashing example from the original discussion of sparse IDs."
      },
      {
        "type": "h2",
        "html": "From vectors to ranking features"
      },
      {
        "type": "p",
        "html": "Real-time personalization itself is not mysterious once the behavior collection system exists. You can aggregate recent clicked listing embeddings, compare them with candidate listings through cosine similarity, and feed those similarities into the ranking model."
      },
      {
        "type": "p",
        "html": "The paper is valuable because it makes the operational details concrete: short-term click embeddings, long-term booking signals, user-type and listing-type embeddings in the same vector space, daily training, offline vector evaluation, and online ranking experiments."
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-04.jpg",
        "alt": "Airbnb offline embedding ranking evaluation",
        "width": 936,
        "height": 714,
        "caption": "Offline ranking evaluation using embedding similarity."
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-05.jpg",
        "alt": "Airbnb user type and listing type embeddings",
        "width": 1036,
        "height": 602,
        "caption": "User-type and listing-type embeddings need to live in the same vector space."
      },
      {
        "type": "h2",
        "html": "The underused signal: negative feedback"
      },
      {
        "type": "p",
        "html": "Recommendation systems tend to overuse positive feedback: clicks, purchases, bookings. But negative feedback is important because one common user complaint is that personalization becomes a pile of similar content. Airbnb’s skipped listings and host rejections are interesting because they give the system a way to learn from what did not work."
      },
      {
        "type": "p",
        "html": "In Mogujie’s scenario, we did not yet get clear gains from this kind of negative feedback, but the direction remains important. The paper is worth reading precisely because it is full of details that can be tried, questioned, and adapted in real systems."
      }
    ],
    "blocksZh": [
      {
        "type": "p",
        "html": "这篇论文是我最近觉得非常值得一读的，写了篇总结放在了专栏，但感觉还是提问的形式更有助于讨论。原文如下："
      },
      {
        "type": "p",
        "html": "Airbnb这篇论文拿了今年KDD best paper，和16年google的W&amp;D类似，并不fancy，但非常practicable，值得一读。可喜的是，据我所知，国内一线团队的实践水平并不比论文中描述的差，而且就是W&amp;D，国内也有团队在论文没有出来之前就做出了类似的结果，可见在推荐这样的场景，大家在一个水平线上。希望未来国内的公司，也发一些真正实用的paper，不一定非要去发听起来fancy的。"
      },
      {
        "type": "p",
        "html": "自从Word2vec出来后，迅速应用到各个领域中，夸张一点描述，万物皆可embedding。在NLP中，一个困难是如何描述词，传统有onehot、ngram等各种方式，但它们很难表达词与词之间的语义关系，简单来讲，即词之间的距离远近关系。我们把每个词的Embedding向量理解成它在这个词表空间的位置，即位置远近能描述哪些词相关，那些词不相关。"
      },
      {
        "type": "p",
        "html": "对于互联网场景，比如电商、新闻，同样的，我们很难找到一个合适表达让计算机理解这些实体的含义。传统的方式一般是给实体打标签，比如新闻中的娱乐、体育、八卦等等。且不说构建一个高质量标签体系的成本，就其实际效果来讲，只能算是乏善可陈。类似NLP，完全可以将商品本身或新闻本身当做一个需要embedding的实体。当我们应用embedding方案时，一般要面对下面几个问题："
      },
      {
        "type": "p",
        "html": "希望Embedding表达什么，即选择哪一种方式构建语料<br>如何让Embedding向量学到东西<br>如何评估向量的效果<br>线上如何使用"
      },
      {
        "type": "p",
        "html": "下面我们结合论文的观点来回答上面问题，水平有限，如有错误，欢迎指出。"
      },
      {
        "type": "h2",
        "html": "希望Embedding表达什么"
      },
      {
        "type": "p",
        "html": "前面我们提了Embedding向量最终能表达实体在某个空间里面的距离关系，但并没有讲这个空间是什么。在NLP领域，这个问题不需要回答，就是语义空间，由现存的各式各样的文本语料组成。在其他场景中，以电商举例，我们会直接对商品ID做Embedding，其训练的语料来至于用户的行为日志，故这个空间是用户的兴趣点组成。行为日志的类型不同，表达的兴趣也不同，比如点击行为、购买行为，表达的用户兴趣不同。故商品Embedding向量最终的作用，是不同商品在用户兴趣空间中的位置表达。"
      },
      {
        "type": "p",
        "html": "很多同学花很多时间在尝试各种word2vec的变种上，其实不如花时间在语料构建的细节上。首先，语料要多，论文中提到他们用了800 million search clicks sessions，在我们尝试Embedding的实践中，语料至少要过了亿级别才会发挥作用。其次，session的定义很重要。word2vec在计算词向量时和它context关系非常大，用户行为日志不像文本语料，存在标点符合、段落等标识去区分词的上下文。"
      },
      {
        "type": "p",
        "html": "举个例子，假设我们用用户的点击行为当做语料，当我们拿到一个用户的历史点击行为时，比如是list(商品A, 商品B，商品C，商品D)，很有可能商品B是用户搜索了连衣裙后点的最后一个商品，而商品C是用户搜索了手机后点击的商品，如果我们不做区分，模型会认为B和C处以一个上下文。"
      },
      {
        "type": "p",
        "html": "具体的session定义要根据自身的业务诉求来，不存在标准答案，比如上面的例子，如果你要做用户跨兴趣点的变换表达，也是可以的，论文中给出了airbnb的规则："
      },
      {
        "type": "p",
        "html": "A new session is started whenever there is a time gap of more than 30 minutes between two consecutive user clicks."
      },
      {
        "type": "p",
        "html": "值得一提的是，论文中用点击行为代表短期兴趣和booking行为代表长期兴趣，分别构建Embedding向量。关于长短期兴趣，业界讨论很多，我的理解是长期兴趣更稳定，但直接用单个用户行为太稀疏了，无法直接训练，一般会先对用户做聚类再训练。"
      },
      {
        "type": "h2",
        "html": "如何让Embedding向量学到东西<br>模型细节"
      },
      {
        "type": "p",
        "html": "一般情况下，我们直接用Word2vec，效果就挺好。论文作者根据Airbnb的业务特点，做了点改造，主要集中在目标函数的细节上，比较出彩。先来看一张图："
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-01.jpg",
        "alt": "Airbnb global context 模型图",
        "width": 982,
        "height": 516,
        "caption": ""
      },
      {
        "type": "p",
        "html": "主要idea是增加一个global context，普通的word2vec在训练过程中，词的context是随着窗口滑动而变化，这个global context是不变的，原文描述如下："
      },
      {
        "type": "p",
        "html": "Both are useful from the standpoint of capturing contextual similarity, however booked sessions can be used to adapt the optimization such that at each step we predict not only the neighboring clicked listings but the eventually booked listing as well. This adaptation can be achieved by adding booked listing as global context, such that it will always be predicted no matter if it is within the context window or not"
      },
      {
        "type": "p",
        "html": "再看下它的公式，更容易理解："
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-02.jpg",
        "alt": "Airbnb global context 目标函数公式",
        "width": 1002,
        "height": 152,
        "caption": ""
      },
      {
        "type": "p",
        "html": "注意到公式的最后一项和前面两项的区别，在累加符号的下面，没有变D限制。我的理解是，word2vec的算法毕竟是非监督的，而Airbnb的业务最终是希望用户Booking，加入一个约束，能够将学到的Embedding向量更好的和业务目标靠近。后面还有一个公式，思路是类似的，不再赘述。"
      },
      {
        "type": "p",
        "html": "这个思路也可以理解成另一种简单的多目标融合策略，另一篇阿里的论文也值得一读，提出了完整空间多任务模型（Entire Space Multi-Task Model，ESMM）来解决。"
      },
      {
        "type": "p",
        "html": "数据稀疏是核心困难"
      },
      {
        "type": "p",
        "html": "Word2vec的算法并不神奇，还是依赖实体出现的频次，巧妇难为无米之炊，如果实体本身在语料中出现很少，也很好学到好的表达。曾经和阿里的同学聊过一次Embedding上线效果分析，认为其效果来源于中部商品的表达，并不是大家理解的长尾商品。头部商品由于数据量丰富，类似i2i的算法也能学的不错，而尾部由于数据太稀疏，一般也学不好，所以embedding技术要想拿到不错的收益，必须存在一批中部的商品。"
      },
      {
        "type": "p",
        "html": "论文中也提到，他们会对entity做个频次过滤，过滤条件在5-10 occurrences。有意思的是，以前和头条的同学聊过这个事情，他们那边也是类似这样的频次，我们这边会大一点。目前没有做的很细致，还未深究这个值的变化对效果的影响，如果有这方面经验的同学，欢迎指出。"
      },
      {
        "type": "p",
        "html": "另一个方法，也是非常常见，即对稀疏的id做个聚类处理，论文提了一个规则，但和Airbnb的业务耦合太深了，其他业务很难直接应用，但可以借鉴思想。阿里以前提过一种sixhot编码，来缓解这个问题，不知道效果如何。也可以直接hash，个人觉得这个会有损，但tensorflow的官网教程上，feature columns部分关于Hashed Column有一段话说是无损的："
      },
      {
        "type": "p",
        "html": "At this point, you might rightfully think: &quot;This is crazy!&quot; After all, we are forcing the different input values to a smaller set of categories. This means that two probably unrelated inputs will be mapped to the same category, and consequently mean the same thing to the neural network. The following figure illustrates this dilemma, showing that kitchenware and sports both get assigned to category (hash bucket) 12:"
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-03.jpg",
        "alt": "Hashed Column 示例图",
        "width": 1224,
        "height": 470,
        "caption": ""
      },
      {
        "type": "p",
        "html": "As with many counterintuitive phenomena in machine learning, it turns out that hashing often works well in practice. That's because hash categories provide the model with some separation. The model can use additional features to further separate kitchenware from sports."
      },
      {
        "type": "h2",
        "html": "离线如何评估效果"
      },
      {
        "type": "p",
        "html": "向量评估的方式，主要用一些聚类、高维可视化tnse之类的方法，论文中描述的思路和我的另一篇文章<a href=\"https://zhuanlan.zhihu.com/p/35491904比较像\">https://zhuanlan.zhihu.com/p/35491904比较像</a>。当Airbnb的工具做的比较好，直接实现了个系统来帮助评估。"
      },
      {
        "type": "p",
        "html": "值得一提的是，论文还提出一种评估方法，用embedding向量做排序，去和真实的用户反馈数据比较，直接引用airbnb知乎官方账号描述："
      },
      {
        "type": "p",
        "html": "更具体地说，假设我们获得了最近点击的房源和需要排序的房源候选列表，其中包括用户最终预订的房源；通过计算点击房源和候选房源在嵌入空间的余弦相似度，我们可以对候选房源进行排序，并观察最终被预订的房源在排序中的位置。"
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-04.jpg",
        "alt": "Airbnb embedding 离线排序评估图",
        "width": 936,
        "height": 714,
        "caption": ""
      },
      {
        "type": "p",
        "html": "上图可以看出，d32 book+neg的效果最好。"
      },
      {
        "type": "h2",
        "html": "线上如何用"
      },
      {
        "type": "p",
        "html": "论文中反复提到的实时个性化并不难，只要支持一个用户实时行为采集的系统，就有很多种方案去实现实时个性化，最简单就是将用户最近的点击序列中的实体Embedding向量做加权平均，再和候选集中的实体做cosine距离计算，用于排序。线上使用的细节比较多，论文中比较出彩的点有两个："
      },
      {
        "type": "h2",
        "html": "多实体embedding向量空间一致性问题"
      },
      {
        "type": "p",
        "html": "这是一个很容易被忽视的问题，当需要多个实体embedding时，要在意是否在一个空间，否则计算距离会变得很奇怪。airbnb在构建long-term兴趣是，对用户和list做了聚类，原文如此描述："
      },
      {
        "type": "p",
        "html": "To learn user_type and listinд_type embeddings in the same vector space we incorporate the user_type into the booking sessions."
      },
      {
        "type": "image",
        "src": "/images/blog/airbnb-real-time-personalization-05.jpg",
        "alt": "Airbnb user type 和 listing type embedding 图",
        "width": 1036,
        "height": 602,
        "caption": ""
      },
      {
        "type": "p",
        "html": "即直接将二者放在一个语料里面训练，保证在一个空间。如此，计算的cosine距离具有实际的意义。"
      },
      {
        "type": "h2",
        "html": "Negative反馈"
      },
      {
        "type": "p",
        "html": "无论是点击行为还是成交行为，都是用户的positive反馈，需要用户付出较大的成本，而另一种隐式的负反馈，我们很少用到（主要是噪音太强）。当前主流的个性化被人诟病最多的就是相似内容扎堆。给用户推相似内容，已经是被广泛验证有效的策略，但我们无法及时有效的感知到用户的兴趣是否已经发生变化，导致损坏了用户体验。因此，负反馈是一个很好的思路，airbnb给出了skipped listing_ids的策略。"
      },
      {
        "type": "p",
        "html": "比较可惜的是，我们目前没有在蘑菇街的场景拿到这个负反馈的收益，如果有同学在相关方面有经验，欢迎来指导。"
      }
    ]
  },
  {
    "slug": "ranking-engineer-first",
    "titleZh": "论算法工程师首先是个工程师之深度学习在排序应用踩坑总结",
    "titleEn": "Algorithm Engineers Are Engineers First",
    "date": "2018-09-12",
    "source": "Zhihu column",
    "sourceUrl": "https://zhuanlan.zhihu.com/p/44315278",
    "excerptZh": "一篇来自蘑菇街搜索推荐阶段的过千赞专栏，用深度学习排序项目踩坑解释为什么算法工程师首先要是工程师。",
    "excerptEn": "A ranking-systems postmortem from the Mogujie period: why algorithm engineers need engineering judgment before model cleverness.",
    "translationMode": "adapted",
    "blocksEn": [
      {
        "type": "p",
        "html": "During campus recruiting, we often interviewed candidates for algorithm roles by starting with one or two simple coding problems before moving into machine-learning questions. Some candidates complained that we were not really asking about machine learning. In many cases, the reason was simple: the code was too weak for the interviewer to keep going."
      },
      {
        "type": "p",
        "html": "Machine-learning roles are easy to romanticize. People imagine deriving formulas and tuning models all day. In reality, our search and recommendation work required a very different kind of ability: understanding systems, data pipelines, serving constraints, performance profiling, and the ugly details that determine whether a model can actually create value online."
      },
      {
        "type": "h2",
        "html": "The first lesson: stop chasing fancy architectures too early"
      },
      {
        "type": "p",
        "html": "In one deep-learning ranking project, we spent months stepping on pits. We were overly attracted to fancy paper ideas and hoped that changing model structures would produce gains. Most of that failed. The meaningful gains came from a more boring place: using more training data, improving sample construction and cleaning, choosing classic model structures carefully, and respecting the optimizer."
      },
      {
        "type": "ul",
        "items": [
          "Deep learning needs much more data than traditional models; increasing sample size can visibly improve results.",
          "At the beginning, forget many of the clever tricks in papers. A workable solution is usually a question of “how much better”, not “whether it works at all”.",
          "Tuning speed matters. If model iteration is alchemy, the faster alchemist has an advantage.",
          "Embeddings are extremely powerful; spend serious effort on how the model represents IDs.",
          "Care about computation. A ranking model eventually has to serve online traffic."
        ]
      },
      {
        "type": "h2",
        "html": "Training is an engineering problem"
      },
      {
        "type": "p",
        "html": "At first, we fed the existing baseline features into deep models: DNN, DFM, LSTM, and so on. The results were worse than logistic regression. Part of the reason was embarrassing but real: to move quickly, we loaded everything into memory, limited the data scale, and handled part of the preprocessing in Python. GPU utilization was unstable because computation kept bouncing between CPU and GPU."
      },
      {
        "type": "p",
        "html": "After profiling, we moved the sample construction to Spark and generated TFRecord data. The whole construction pipeline became nearly ten times faster than the old Hive SQL plus HDFS-to-local process, and we could use much more data. The model improved. This was not a modeling miracle; it was engineering work."
      },
      {
        "type": "h2",
        "html": "Industrial embeddings are not NLP toy sizes"
      },
      {
        "type": "p",
        "html": "In NLP papers, hundreds of thousands of words can be called large-scale. In industrial recommender systems, user IDs and item IDs easily reach millions or tens of millions. Embedding lookup can run out of memory quickly. Hashing, SimHash, metadata-based recoding, and selective treatment of sparse IDs all become practical design choices."
      },
      {
        "type": "p",
        "html": "Wide & Deep also brings sparse-model training problems. Many implementations use dense tensors. Once feature scale reaches hundreds of millions, parameter-server communication becomes a disaster. Reading the TensorFlow source and using sparse ops can make a large difference. Sometimes the elegant solution is simply understanding the system deeply enough."
      },
      {
        "type": "h2",
        "html": "Online ranking is not batch prediction"
      },
      {
        "type": "p",
        "html": "In training, prediction code is often organized by batch size. Online ranking is different. When a user arrives, the system may need to rerank thousands of candidates. If user-side features are copied thousands of times and TensorFlow performs thousands of embedding lookups, latency will be terrible. Doing user-side lookups once at the beginning and then copying memory can dramatically reduce response time."
      },
      {
        "type": "p",
        "html": "Other costs hide in attention modules, cross-network implementations, and small algebraic choices. A simple use of commutativity in DCN’s cross layer can bring a large performance improvement. These are not separate from algorithm work; they are part of making the algorithm real."
      },
      {
        "type": "h2",
        "html": "Theory still matters"
      },
      {
        "type": "p",
        "html": "None of this means machine-learning theory is unimportant. Theory gives you belief and direction when a project takes a year, when there is no clear intermediate output, and when the business keeps asking for KPIs. But beautiful assumptions meet cruel real-world data very quickly. Moving toward truth requires both theoretical understanding and the ability to dirty your hands in the system."
      },
      {
        "type": "p",
        "html": "The relationship is simple: theory points the way; engineering is the blade that cuts through the road. Without theory, you have no direction. Without coding ability, you can only watch from the side. Algorithm engineers are engineers first."
      }
    ],
    "blocksZh": [
      {
        "type": "p",
        "html": "引子"
      },
      {
        "type": "p",
        "html": "最近校招面试到吐，算法岗位有点太热了，简直心力憔悴。我们的面试分两个部分，先是做一两道编码题，然后才是考察机器学习的知识。很多同学不理解，在网上diss我们，说什么机器学习基本没有问。这种情况，一般是代码做的太烂了，面试官都没有兴趣去了解机器学习部分。"
      },
      {
        "type": "p",
        "html": "机器学习算法岗位，很容易让大家有个误解，认为平时工作就是推推公式，调调参数。鉴于此，本文借用下我们团队最近的一个重要项目：深度学习在搜索、推荐中的应用，来描述下平时我们是怎么干活的，看完之后，大家应该很容易理解为何我们要求有编码能力。"
      },
      {
        "type": "p",
        "html": "其实，我们的编码题真的很简单，不用刷题也能做出来，看看其他公司出的题，已经有点类似面试造原子弹，进来卖茶叶蛋的蜜汁感觉。当然，他们有资本，可以通过这种方式选到很聪明的候选人。"
      },
      {
        "type": "p",
        "html": "回到正题，我们从去年年底开始探索深度学习在搜索、推荐中的应用，包括排序和召回。以前我们常常用和工程同学合作，对系统的理解，比如推荐引擎、搜索引擎来表达编码能力的重要性，可能对于应届生来讲，有点模糊。这次的项目经历可能更好一些。"
      },
      {
        "type": "h2",
        "html": "先总结下指导思想"
      },
      {
        "type": "p",
        "html": "这大半年，我们踩了很多坑，特别是痴迷论文中的各种fancy结构，寄希望于换换模型拿到收益。最终都纷纷被打脸，反而是回归到开始，从使用更多的样本数据，改善样本清洗、构造的逻辑，谨慎选择经典的模型结构，对优化算法保持敬畏等等，拿到了不错的收益。先来几点务虚的鸡汤，大概有以下几点："
      },
      {
        "type": "p",
        "html": "对比传统模型，深度学习更需要大量的数据去学习，样本数据的增加能明显的改善模型的结果。<br>在初期，请忘记paper里面各式各样的奇技淫巧。一套有效的方案，其效果是多和少的问题，不是有和无的问题。好好调参，比乱试各种论文idea有效。<br>深度学习真的可以自称调参炼丹师，所以比别人试的更快，是炼丹师的核心竞争力。<br>Embedding太神奇，请把主要精力花在这里，深度模型对id的理解可以震惊到你。<br>关心你的模型的计算效率，最终还是要上线的，绕不过去的性能问题。"
      },
      {
        "type": "h2",
        "html": "训练中的工程能力篇，就是各种踩坑各种填坑<br>样本规模的问题"
      },
      {
        "type": "p",
        "html": "一开始，我们把现有基线的特征数据喂到了深度模型中，试过dnn、dfm、lstm等等，发现效果比lr还差。当时为了快速尝试，将所有的数据load到了内存，限制了数据规模，而且有部分数据预处理的工作也是在python中处理，导致计算在cpu和gpu之间频繁切换，gpu利用率抖动很厉害。基于tf提供的性能工具，做了点分析后，判断是特征预处理这部分移太耗时了。另外，模型的参数很大，但是样本数不够，急需增加样本量。我们用spark将样本数据构造成tfrecord的格式，整个构建过程对比原来基于hive sql，再从hdfs拉到本地，快了近10倍，而且能用的样本数据量大了很多，发现模型效果好了很多。"
      },
      {
        "type": "h2",
        "html": "embedding id量级过大的问题"
      },
      {
        "type": "p",
        "html": "深度学习是在图像、语音等场景起家，经常在nlp的论文中，将几十万的word做embedding称为大规模。工业界做user和item embedding的同学应该笑了。userid和itemid非常容易过百万、千万的量级，导致生成embedding lookup oom。可以参考我上篇文章：<a href=\"https://zhuanlan.zhihu.com/p/39774203\">https://zhuanlan.zhihu.com/p/39774203</a>。"
      },
      {
        "type": "p",
        "html": "有些公司会选择对id进行hash，再做embedding，比如tf的官网就建议这样：<a href=\"https://www.tensorflow.org/guide/feature_columns#hashed_column\">https://www.tensorflow.org/guide/feature_columns#hashed_column</a>。也有些会选择simhash来替换直接hash。我们目前能做百万级别的原始id，后续如果需要加大量级，更倾向于只对样本特别稀疏的id做hash或根据id的metadata做重编码来做。"
      },
      {
        "type": "h2",
        "html": "Wide模型带来的稀疏模型训练问题"
      },
      {
        "type": "p",
        "html": "大部分的wide &amp; deep代码实现，其实用的tensor都是dense的。tf基于PS做的模型训练，当你的特征规模在亿级别时，网络通信是个灾难，加上grpc的垃圾性能，网卡利用率上不去，训练的时间大部分都耗在通信上了。"
      },
      {
        "type": "p",
        "html": "但如果花点心思看看tf的源码，解决方法其实很简单，采用一些sparse的op就行。比如用sparse_gather，就能解决网络传输的问题。但这个不是彻底的解决方案，tf在计算的时候又会把sparse的tensor转成dense做。继续看看源码，会发现tf自身实现的embedding_lookup_sparse。换个角度来理解，天然就能支持sparse的wide模型训练。把sparse的wide模型理解成embedding size为1的情况，上层接个pooling做sum，就是我们要的wide的output结果，方案很优雅。"
      },
      {
        "type": "p",
        "html": "分布式下训练速度不能随着batch size增加变快"
      },
      {
        "type": "p",
        "html": "这个问题，单纯看性能分析还不好发现。还是去看下TF的代码实现，其实是TF默认有个dimension压缩的优化带来的。TF为了节省存储，会对一个batch内的相同的feature做hash压缩，这里会有个distinct的操作，在batch size大的时候，性能损耗很明显。改下参数，就可以取消该操作，不好的地方是浪费点内存。"
      },
      {
        "type": "p",
        "html": "还有两个核心问题：TF不支持sparse模型和分布式下work的checkpoint问题，这里不展开了。"
      },
      {
        "type": "h2",
        "html": "线上性能篇：<br>真实线上场景与batch size的训练的差异"
      },
      {
        "type": "p",
        "html": "真实排序的时候，一个用户过来，需要精排的候选集可能有几千。而我们在训练的时候，基于batchsize方式组织的predict代码。会将用户侧的feature复制几千次，变成一个矩阵输入到模型中。如果给tf自己做，这里就会有几千次的embedding lookup，非常的耗时。如果我们选择在请求的一开始，就把用户侧的lookup做掉，然后去做点内存复制，就能大大减少rt。"
      },
      {
        "type": "p",
        "html": "另外一个耗时大头是attention，这个解决方案也很多，比如用查表近似就可以。"
      },
      {
        "type": "p",
        "html": "还有一些是模型实现的细节不好导致性能很差，比如DCN的cross实现，一个简单的交换律能带来巨大的性能提升，参考：<a href=\"https://zhuanlan.zhihu.com/p/43364598\">https://zhuanlan.zhihu.com/p/43364598</a>"
      },
      {
        "type": "p",
        "html": "扯淡开始"
      },
      {
        "type": "p",
        "html": "上面很多工作，都是算法工程师和工程同学一起深入到代码细节中去扣出来的，特别是算法工程师要给出可能的问题点。做性能profile，工程的同学比我们在行，但是模型中可能的性能问题，我们比他们了解的多。当然也有很多同学diss，上面这些都是工程没有做好啊，工程好了不需要关心。但是，真正的突破必然是打破现有的体系，需要你冲锋陷阵的时候自己不能上，别人凭什么听你的，跟你干。大概率就是在后面维护点边缘业务了。"
      },
      {
        "type": "h2",
        "html": "难道机器学习理论不重要吗"
      },
      {
        "type": "p",
        "html": "当然不是，这篇已经写得太长了，只讲两个点。"
      },
      {
        "type": "p",
        "html": "信念的来源：这个其实是很重要的，一个项目，搞个一年半载的，中间没有什么明确的产出，老板要kpi，旁边的同事刷刷的出效果，靠什么支持你去坚持继续填坑，只有对理论认知的信念。<br>假设总是很美好，现实数据很残酷，左脸打完打右脸，啪啪啪的响。怎么一步步的接近真实，解决问题，靠的还是对理论的理解，特别是结合业务的理论理解。"
      },
      {
        "type": "p",
        "html": "工程和理论的关系就有点像，理论起到是指路者的作用，而工程是你前进道路上披荆斩棘的利刃。没有理论就没有方向，没有编码能力，就只能当个吃瓜群众，二者缺一不可。"
      },
      {
        "type": "h2",
        "html": "最后，总结下：算法工程师首先是个工程师。"
      },
      {
        "type": "p",
        "html": "PS：Don’t panic！Make your hands dirty！编码没有那么难。"
      }
    ]
  }
] as const;
