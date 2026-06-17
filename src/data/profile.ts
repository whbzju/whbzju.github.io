export const profile = {
  name: "Wujia / Haibo Wu",
  mark: "WH",
  eyebrow: "AI product builder · Open-source operator · Writer",
  headline: "I build and write about AI products that have to survive real workflows.",
  summary:
    "I work across commercial AI SaaS, open-source agents, creative workflows, and personal memory. Recent work centers on two products: WeShop for global e-commerce creative production, and Memex for local-first personal AI memory.",
  chinese:
    "最近几年，我主要关注 AI 应用如何从 demo 进入真实业务、数据、工作流和长期信任。",
  links: {
    github: "https://github.com/whbzju",
    linkedin: "https://www.linkedin.com/in/haibo-wu-97b60759/",
    zhihu: "https://www.zhihu.com/people/wu-hai-bo",
    memex: "https://github.com/memex-lab/memex",
    weshop: "https://www.weshop.ai/",
  },
};

export const nav = [
  { label: "Work", href: "/works/" },
  { label: "Thesis", href: "/thesis/" },
  { label: "Blog", href: "/blog/" },
  { label: "Events", href: "/events/" },
  { label: "Contact", href: "/contact/" },
];

export const works = [
  {
    title: "Memex",
    type: "Open-source AI product",
    href: "https://github.com/memex-lab/memex",
    image: "/images/memex-memory.webp",
    summary:
      "A local-first AI journal for capturing text, screenshots, voice, and images, then turning them into timeline cards, insights, and personal memory.",
    thesis:
      "Personal agents need a trust layer before they can handle private context over time.",
    proof: ["Open source", "Local-first", "AI journal", "Personal agents"],
  },
  {
    title: "WeShop AI",
    type: "Commercial AI product",
    href: "https://www.weshop.ai/",
    image: "/images/weshop-workflow.webp",
    summary:
      "An AI image and video generation platform for global e-commerce merchants, brands, and creators working with product, model, and advertising assets.",
    thesis:
      "Vertical AI SaaS needs workflow depth, quality control, and distribution, not just model access.",
    proof: ["Vertical AI SaaS", "Creative workflow", "Global users", "Media covered"],
  },
];

export const principles = [
  {
    title: "AI applications must enter workflows",
    text:
      "The valuable layer is not a prettier model interface. It is the repeated task, data structure, collaboration loop, and delivery expectation around the model.",
  },
  {
    title: "Vertical AI SaaS compounds through context",
    text:
      "As model capability becomes widely available, durable product advantage shifts toward domain data, workflow closure, distribution, and operational taste.",
  },
  {
    title: "Personal agents require local-first trust",
    text:
      "When an agent touches screenshots, voice, private notes, and long-term memory, ownership and controllability become product foundations.",
  },
  {
    title: "Open-source growth starts with a real problem",
    text:
      "A small project can still be discovered when the problem is concrete, the progress is visible, and the community can participate in the work.",
  },
];

export const writing = [
  {
    title: "一个没资源的开源项目，是怎么被人看见的",
    source: "Zhihu column",
    href: "https://zhuanlan.zhihu.com/p/2046314257738887994",
    note: "Memex 的开源冷启动复盘：没有预算、没有合作，如何通过正确讨论和真实作品被看见。",
  },
  {
    title: "我们做了一个关于“记录自己”的东西，然后决定把它开源",
    source: "Zhihu column",
    href: "https://zhuanlan.zhihu.com/p/2016169349749166998",
    note: "解释 Memex 的产品理念：AI 时代，个人记录、隐私和长期信任为什么重新变重要。",
  },
  {
    title: "我们发布了WeShop商拍1.5版----分享一些最近的思考",
    source: "Zhihu column",
    href: "https://zhuanlan.zhihu.com/p/680507675",
    note: "关于 WeShop 产品演进、AI 商拍场景和电商创意工作流的阶段性思考。",
  },
];

export const events = [
  {
    title: "WeShop唯象总经理吴海波：AI创业已非“套壳应用”时代",
    source: "36Kr talk",
    href: "https://m.36kr.com/p/3262118027476743",
    note: "关于 AI 应用创业、垂直场景和商业化产品深度的公开分享。",
  },
  {
    title: "AI Entrepreneurship Era Is No Longer About Wrapper Apps",
    source: "36Kr Global",
    href: "https://eu.36kr.com/en/p/3262118027476743",
    note: "English coverage of the same product thesis for an international audience.",
  },
  {
    title: "我们是独立于蘑菇街的产品，改老的东西比做新东西更难",
    source: "Tencent News",
    href: "https://news.qq.com/rain/a/20231113A04U5B00",
    note: "关于 WeShop、产品独立性、组织和业务演进的访谈。",
  },
  {
    title: "Sora 爆火将如何改变广告行业？",
    source: "36Kr",
    href: "https://m.36kr.com/p/2674977768683266",
    note: "围绕视频生成模型、广告行业和商业创意工作流的讨论。",
  },
  {
    title: "聊个小众一点的方向，AI 团队的组织方式",
    source: "Podcast",
    href: "https://www.zhihu.com/pin/1934944940464518617",
    note: "一次关于 AI 团队组织、产品节奏和创业判断的公开播客讨论。",
  },
];

export const writingCollections = [
  {
    title: "Zhihu columns and answers",
    description:
      "A long-running thinking trail across AI products, e-commerce creative workflows, agents, and open-source growth.",
    items: [
      ["Zhihu column", "和大家汇报下我们电商AI模特产品WeShop beta版本开放测试", "https://zhuanlan.zhihu.com/p/629144911"],
      ["Zhihu column", "我们发布了WeShop商拍1.5版----分享一些最近的思考", "https://zhuanlan.zhihu.com/p/680507675"],
      ["Zhihu answer", "AI 技术代替电商模特，现在可以实现了吗？", "https://www.zhihu.com/question/590884963/answer/2983851310"],
      ["Zhihu answer", "怎么成为一个 AI agent 工程师？", "https://www.zhihu.com/question/1936375725931361485/answer/1951304824327996800"],
    ],
  },
];

export const contactReasons = [
  {
    label: "Product",
    title: "AI applications beyond demos",
    text:
      "Products that need to enter real workflows, data loops, business constraints, and repeated use.",
  },
  {
    label: "Open source",
    title: "Agents, memory, and personal data",
    text:
      "Local-first design, private context, long-term memory, and personal agents.",
  },
  {
    label: "Business",
    title: "Global AI SaaS and commerce",
    text:
      "Creative workflows, e-commerce media generation, global SaaS distribution, and commercialization.",
  },
];
