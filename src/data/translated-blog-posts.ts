import zhPosts from "./translated-blog-posts.zh.json";

const zhBySlug = Object.fromEntries(zhPosts.map((post) => [post.slug, post]));

const img = (src, alt, width, height, caption = "") => ({
  type: "image",
  src,
  alt,
  width,
  height,
  caption,
});

const withEnglish = (slug, english) => ({
  ...zhBySlug[slug],
  ...english,
  translationMode: "adapted",
});

export const translatedBlogPosts = [
  withEnglish("weshop-aigc-product-reflections", {
    titleEn: "What Building WeShop Taught Me About Generative-AI Products",
    excerptEn:
      "Reflections from the first commercial release of WeShop: how to define a product before the technology is mature, and why compute cost, evaluation, and real customer feedback become the core constraints.",
    blocksEn: [
      {
        type: "p",
        html: "When we opened the WeShop beta on May 10, both growth and user feedback exceeded what our small team expected. It felt a little like the early mobile-internet years: if the product solved a real problem, users and media would carry part of the distribution for you.",
      },
      {
        type: "p",
        html: "The formal release also made one thing clear: WeShop would be a paid product built around AI compute. That sounds simple, but for an AIGC product it changes almost every product decision, from onboarding to infrastructure.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-01.png",
        "WeShop early product reference",
        1472,
        580,
        "Early WeShop product context from the original Zhihu post."
      ),
      img(
        "/images/blog/weshop-aigc-product-reflections-02.jpg",
        "WeShop early product article screenshot",
        2442,
        3200,
        "Related WeShop background material from the original post."
      ),
      {
        type: "p",
        html: "A lot of AI discussion at the time was philosophical. We were more interested in the practical question: what breaks when you try to turn image-generation technology into a real commercial workflow?",
      },
      {
        type: "h2",
        html: "The first problem is product definition before the technology is fully ready",
      },
      {
        type: "p",
        html: "Every AI application faces the same tension. If the technology is already mature, the opportunity may be gone. If the technology is not mature, the product can easily be worse than the old workflow. That is why product judgment becomes even more important in the AI era.",
      },
      {
        type: "p",
        html: "For WeShop, the obvious research direction had always been virtual try-on: give the system a product photo and have it place the product naturally on many different models. Mogujie had explored this since it first had a computer-vision team. My view then was that the technology still did not meet the minimum bar for a B2B workflow, though it could work earlier in some consumer-facing experiments.",
      },
      {
        type: "p",
        html: "A better route is often to combine technical taste with business sense: choose a promising technical direction, then apply it to a business problem you have understood for years.",
      },
      {
        type: "p",
        html: "Once the technical route and interaction are good enough, users become more imaginative than the team. One customer gave us a very concrete lesson.",
      },
      {
        type: "p",
        html: "In the mannequin workflow, merchants often uploaded hollow mannequin photos without heads or limbs. These images are common in the industry, but they are difficult for a model-generation system.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-03.jpg",
        "Original mannequin product photo",
        1118,
        750,
        "A common mannequin-style product input."
      ),
      {
        type: "p",
        html: "If used directly, the result usually failed in obvious ways.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-04.jpg",
        "Unstable WeShop generation from mannequin input",
        1149,
        1107,
        "A direct generation result before adapting the workflow."
      ),
      {
        type: "p",
        html: "We spent a lot of time trying to make these inputs produce natural model photos. Then one customer found a workaround: add a few rough strokes before running the generation.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-05.jpg",
        "Customer workaround with rough drawing",
        769,
        1023,
        "A simple user-created hint on top of the mannequin image."
      ),
      {
        type: "p",
        html: "With that hint, the output improved dramatically.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-06.jpg",
        "Improved WeShop output after user workaround",
        1148,
        759,
        "The same workflow after a user-provided hint."
      ),
      {
        type: "p",
        html: "Other customers quickly extended the trick. A product weakness became partly solvable through a workflow pattern we had not designed ourselves.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-07.jpg",
        "Further customer-developed workflow example",
        1080,
        1439,
        "Users pushed the workflow further than the team expected."
      ),
      {
        type: "h2",
        html: "Compute cost becomes a product constraint",
      },
      {
        type: "p",
        html: "WeShop was the first product in my career where compute cost became a serious concern at such an early user scale. GPUs were expensive, hard to procure, and often needed urgently because growth was difficult to forecast.",
      },
      {
        type: "ol",
        items: [
          "Different customer requests needed different combinations of base models, LoRAs, ControlNets, and traditional CV models. These models are large, GPU utilization is high during each job, and cold-starting them online is too slow for users. We had to pre-configure resource pools, which created waste and occasional queues.",
          "The output quality required constant parameter tuning and offline training. Because of cost constraints, online and offline workloads had to share infrastructure, and our switching logic was not yet intelligent enough.",
          "Different GPU types across different data centers made resource management even harder.",
        ],
      },
      {
        type: "p",
        html: "For an AIGC business, compute is not a back-office detail. It is part of the life line of the product.",
      },
      {
        type: "h2",
        html: "Evaluation is still unsolved",
      },
      {
        type: "p",
        html: "Controlling a diffusion model to commercial quality is hard. In one iteration, the left sleeve edge clearly improved after we trained a more targeted base model.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-08.jpg",
        "Model evaluation comparison",
        1440,
        684,
        "A targeted model iteration improved one visible product-detail problem."
      ),
      {
        type: "p",
        html: "But we soon found that the model had overfit. In other scenes, the generated image acquired a hazy layer that damaged the texture and commercial quality of the photo.",
      },
      img(
        "/images/blog/weshop-aigc-product-reflections-09.jpg",
        "Overfitting artifact in commercial generation",
        1440,
        960,
        "An improvement in one case created quality regressions in another."
      ),
      {
        type: "p",
        html: "The same pattern appeared across base models, LoRAs, templates, and parameter choices. Even with strong community developers on the team, there was no obvious best practice for deciding whether a model change was truly better.",
      },
      {
        type: "p",
        html: "That is why real customer demand matters so much. AIGC teams have to improve together with customers inside an uncertain experience space.",
      },
      {
        type: "h2",
        html: "Small teams, hard choices",
      },
      {
        type: "p",
        html: "WeShop was an AI product under Mogujie, run by a small, self-driven team distributed across several regions. We were hiring for frontend, backend, and algorithm roles, but we also wanted to stay small.",
      },
      {
        type: "p",
        html: "The frontend challenge was designing interactions around AI. The backend stack was mainly Go and Python, with heavy work around compute management. The algorithm work required familiarity with Stable Diffusion, LLMs, and strong coding ability; traditional CV and NLP backgrounds were useful but not enough by themselves.",
      },
      {
        type: "p",
        html: "Many people reached out about APIs, private deployments, ecosystem integrations, and partnerships. With limited resources, we had to queue many requests and make tradeoffs. The practical lesson was simple: in early AI productization, focus is not a slogan. It is the only way a small team survives.",
      },
    ],
  }),
  withEnglish("record-yourself-often", {
    titleEn: "Why Record Yourself So Often?",
    excerptEn:
      "A public-facing version of the Memex thesis: in the AI era, personal fragments, private context, and long-term trust become more valuable, not less.",
    blocksEn: [
      {
        type: "p",
        html: "We are living through a strange historical moment. AI is rewriting the rules at an extraordinary speed: writing, drawing, coding, analyzing. Almost every day it pushes against abilities humans used to be proud of.",
      },
      {
        type: "p",
        html: "That creates anxiety: what should I learn next? What is still mine? Where does my value come from?",
      },
      {
        type: "p",
        html: "But precisely because of that, one thing becomes more important: recording the traces of your own life.",
      },
      {
        type: "p",
        html: "Not because recording is efficient. Not because it helps you build a disciplined persona. Because when more external capabilities are handled by AI, the emotions, moments, confusions, and small tremors that belong only to you become harder to replace.",
      },
      {
        type: "h2",
        html: "The problem with diaries is not input",
      },
      {
        type: "p",
        html: "More people are telling others to keep a diary. The advice is not wrong, but the form is heavy. A diary usually means you live through a day, sit down later, recall, filter, organize, and turn it into a coherent paragraph.",
      },
      {
        type: "p",
        html: "That is compression. It preserves the version you have already organized, not life itself.",
      },
      {
        type: "p",
        html: "Real life is often made of fragments: a photo taken without much thought, a sentence you suddenly want to write down, a mood shift you cannot explain, a sadness that arrives late at night.",
      },
      {
        type: "p",
        html: "These fragments look small, but they are often closer to the real you than any polished summary.",
      },
      {
        type: "h2",
        html: "Recording used to feel lonely",
      },
      {
        type: "p",
        html: "Most recording products were hard to keep using not because input was too inconvenient, but because nothing happened afterward. You wrote a sentence, took a photo, saved a thought, and then it disappeared into a timeline.",
      },
      {
        type: "p",
        html: "There was no response, no connection, and no sense of being understood. Over time, recording became a lonely act.",
      },
      {
        type: "p",
        html: "AI changes this for the first time. After you record something, the system can respond. It does not need to be deep every time. Often the user just needs a light, natural response that catches the moment.",
      },
      {
        type: "p",
        html: "Over long-term use, connections begin to appear between fragments. Patterns you would not have noticed start to surface. At that point, recording is no longer just “I saved something.” It becomes: I am slowly understanding who I am through these fragments.",
      },
      {
        type: "h2",
        html: "What Memex is trying to do",
      },
      {
        type: "p",
        html: "Memex is not a traditional diary app, and it is not simply an emotional-companion product. It is an attempt at a different kind of personal record: not just storing life, but helping you see yourself through it.",
      },
      {
        type: "ul",
        items: [
          "Recording should be almost frictionless. Memex does not ask you to change your habits or write long entries every day. A sentence, a photo, a voice note: record the way you already record.",
          "AI should work in the background. Multiple agents can organize records, generate cards, extract insights, and connect memories without making the user do that labor.",
          "The response should feel real. Insights should not become reports, and companionship should not become marketing copy. The goal is that one day you look back and feel: this is what I was like.",
        ],
      },
      {
        type: "h2",
        html: "Why open source matters here",
      },
      {
        type: "p",
        html: "Memex moved everyone who worked on it, but we also saw the difficulty clearly. Inside a commercial company, it is hard to maintain a product that is fundamentally about the person rather than traffic or monetization.",
      },
      {
        type: "p",
        html: "Model costs are high. Operating costs are high. Under growth and profit pressure, we could not simply promise that the original idea would remain intact forever.",
      },
      {
        type: "p",
        html: "There is also a trust problem. Because on-device models are not yet strong enough, Memex still has to call cloud models in some scenarios. We do not want to pretend that this problem has already been solved.",
      },
      {
        type: "p",
        html: "So we decided to open source it. The most sensitive data deserves the highest level of trust. A system that touches your emotions, relationships, vulnerability, and personal confusion should not rely only on one company's promise. It should be transparent, inspectable, and guarded by a community.",
      },
      img(
        "/images/blog/record-yourself-often-01.jpg",
        "Memex demo video still",
        894,
        1920,
        "A simple Memex demo from the original answer."
      ),
      img(
        "/images/blog/record-yourself-often-02.jpg",
        "Memex product screenshot",
        1080,
        2346
      ),
      img(
        "/images/blog/record-yourself-often-03.jpg",
        "Memex product screenshot",
        1080,
        2400
      ),
      img(
        "/images/blog/record-yourself-often-04.jpg",
        "Memex product screenshot",
        1179,
        2556
      ),
      img(
        "/images/blog/record-yourself-often-05.jpg",
        "Memex product screenshot",
        1179,
        2556
      ),
      img(
        "/images/blog/record-yourself-often-06.jpg",
        "Memex product screenshot",
        1080,
        2346
      ),
    ],
  }),
  withEnglish("chatgpt-application-technology", {
    titleEn: "ChatGPT's Core Technologies, Seen From the Application Layer",
    excerptEn:
      "A 2023 application-builder's reading of ChatGPT: emergence, alignment, prompting, in-context learning, hallucination, multimodality, and the coming wave of AI-native applications.",
    blocksEn: [
      {
        type: "p",
        html: "This piece came from a period of intense discussion around large language models. Rather than write another model-centric explainer, I wanted to look at ChatGPT from the position I knew better: someone who had worked across deep learning, engineering, and internet products.",
      },
      {
        type: "p",
        html: "Large language models did not appear out of nowhere. To understand ChatGPT, you have to trace roughly a decade of deep-learning progress, especially in natural language processing. But for application builders, the more important question is not only how the model works. It is why the product suddenly works.",
      },
      {
        type: "h2",
        html: "Start with the strange phenomenon of emergence",
      },
      {
        type: "p",
        html: "Deep learning has always been criticized for weak theory. But in the history of science, useful applications often arrived before complete theory. Major unexplained phenomena are sometimes the opening for new theory.",
      },
      img(
        "/images/blog/chatgpt-application-technology-01.jpg",
        "Emergent ability chart for large language models",
        1013,
        487,
        "The original post used emergence as the entry point for understanding LLMs."
      ),
      {
        type: "p",
        html: "The mysterious part of LLMs is emergent ability. Researchers still do not have a fully satisfying explanation for why performance can jump sharply after scale passes certain thresholds. If we understood this well, it might even reshape the old debate between statistical and symbolic views of intelligence.",
      },
      {
        type: "p",
        html: "LeCun's criticism of ChatGPT-as-AGI came from a familiar position: statistical methods should not be enough for general intelligence. Yet emergence hangs over that argument like a cloud. If similar effects appeared in vision or multimodal systems, the implications would be much larger than a commercial race between model providers.",
      },
      {
        type: "p",
        html: "There are several possible explanations. Maybe our evaluation metrics are too discontinuous, and the capability is present earlier than the score suggests. Maybe some knowledge and reasoning patterns are learned incorrectly at small scale, then corrected only when the model becomes large enough. Or maybe scale really does produce a qualitative change in a sufficiently complex learned distribution.",
      },
      img(
        "/images/blog/chatgpt-application-technology-02.jpg",
        "U-shaped scaling behavior illustration",
        1440,
        733,
        "A figure from the original post discussing non-smooth capability curves."
      ),
      {
        type: "h2",
        html: "Alignment is the key to productization",
      },
      {
        type: "p",
        html: "Compute, data, and algorithms explain much of AI progress, but productization depends heavily on alignment. This may be one of the places where OpenAI was ahead of the industry in product judgment.",
      },
      {
        type: "p",
        html: "I am using “alignment” here in a practical product sense: the methods that make a model's latent capability line up with what users actually want. Prompting, in-context learning, chain-of-thought style interaction, and RLHF all belong to this broader product story.",
      },
      {
        type: "h2",
        html: "Prompting as the UI/UX of the AI era",
      },
      {
        type: "p",
        html: "Many people compared prompts to a new kind of UI/UX. That comparison is useful. Prompting was not merely a research trick for matching downstream tasks to pretraining. It became a way for users to expose and steer model capabilities.",
      },
      {
        type: "p",
        html: "In-context learning looked at first like a way to distinguish zero-shot use from meta-learning. But later work showed something more surprising: even wrong examples sometimes did not hurt performance much, while examples from the wrong distribution did. The prompt was not simply a label. It was a context-setting interface.",
      },
      img(
        "/images/blog/chatgpt-application-technology-03.jpg",
        "In-context learning illustration",
        1440,
        713,
        "Prompting and in-context learning became part of the product interface."
      ),
      {
        type: "p",
        html: "The same applies to chain-of-thought style prompting. A model that seems weak at reasoning can improve noticeably when the interaction asks it to proceed step by step. Before we understand the mechanism deeply, AI researchers often look like alchemists: trying different spells to summon capability from a system we do not fully understand.",
      },
      {
        type: "p",
        html: "ChatGPT found a better alignment surface: GPT-3.5 plus RLHF, wrapped in a dialogue product. That does not mean the full capability of LLMs has been unlocked. It means interaction design became part of the model's effective intelligence.",
      },
      {
        type: "h2",
        html: "Will LLMs look like search engines or cloud computing?",
      },
      {
        type: "p",
        html: "One important business question is whether large models will resemble Google-style search dominance or AWS-style infrastructure competition. My intuition is closer to the AWS analogy: one company may lead, but multiple strong providers can still exist.",
      },
      {
        type: "p",
        html: "Search and recommendation systems did not truly understand content. They mined user behavior and distribution feedback. In the LLM era, models begin to understand and generate content itself. That weakens some old supply-side moats and changes how distribution may work.",
      },
      {
        type: "p",
        html: "ChatGPT was the best product at the time and had strong user feedback, but the underlying LLM technology was not locked inside OpenAI. Google and Meta also had users, talent, and infrastructure. It was reasonable to expect serious competitors.",
      },
      {
        type: "h2",
        html: "Hallucination and external memory",
      },
      {
        type: "p",
        html: "ChatGPT can produce factual errors, and model training has a time boundary. In production, it can expand the capability boundary of professionals, but it cannot simply replace expertise. The model was especially strong in technical domains partly because the web contains a large amount of high-quality programming and IT material.",
      },
      {
        type: "p",
        html: "After GPT-3, much work studied how models store, modify, and correct knowledge. Some research viewed the transformer's feed-forward layers as a kind of key-value memory. Other work tried to update specific facts through constrained optimization without damaging unrelated knowledge.",
      },
      {
        type: "p",
        html: "From an energy and system-design perspective, I think LLMs should rely less on memorizing every factual detail and more on reasoning over external knowledge. Retrieval-augmented approaches, DeepMind's RETRO, vector databases, LangChain, GPTIndex, and the early new Bing all pointed in that direction.",
      },
      {
        type: "h2",
        html: "When will multimodality really arrive?",
      },
      {
        type: "p",
        html: "I believe multimodal large models are a prerequisite for AGI. Humans learn in a physical world; text is already an abstraction. Vision provides stronger anchors in physical regularities, which may help models learn more fundamental concepts.",
      },
      {
        type: "p",
        html: "But the path is not straightforward. CLIP was useful, but more like BERT than GPT-3. ViT was promising, but the tokenization problem in vision is different from language: text tokens carry semantic structure in a way image patches do not. This is part of why diffusion models became so effective in image generation while transformer-style sequence modeling faced different constraints.",
      },
      {
        type: "p",
        html: "Another guess: truly large multimodal models will need sparsity. If we loosely compare parameter scale with human synapses, GPT-3 was still far smaller. Scaling further while keeping inference cost manageable likely requires sparse architectures and new infrastructure.",
      },
      {
        type: "h2",
        html: "A new era of application innovation",
      },
      {
        type: "p",
        html: "Media often ask which jobs will be affected by AI. The better question may be: which ones will not? I was not fully optimistic about AGI, but the capabilities shown by ChatGPT and diffusion models were broad enough that most industries should take them seriously.",
      },
      {
        type: "p",
        html: "This wave changes human-computer interaction. We will see a generation of applications whose primary interface is natural language. For the first time, machines can interpret human intent with this level of detail, across multiple rounds, with each interaction shaped by context.",
      },
      {
        type: "p",
        html: "Think about Office, Photoshop, or video editing tools. Learning them often means learning a graphical programming language for instructing a computer. If users can express intent directly in natural language, many categories of software can be rebuilt.",
      },
      {
        type: "p",
        html: "That does not mean every AI application will succeed. Every technology has boundaries, and we do not yet know where they are. Many early AI apps will simply wrap an API without a durable moat. The deeper opportunities are in workflow design, alignment with user intent, and surviving long enough for the next maturity cycle.",
      },
      {
        type: "h2",
        html: "Other questions worth tracking",
      },
      {
        type: "ul",
        items: [
          "Data quality is often misunderstood. Many people still think model training mainly requires labeled data, while the NLP scaling story depended heavily on self-supervised objectives such as masked language modeling.",
          "Compute still matters. GPUs face physical limits too, but parallel workloads have allowed rapid growth. In the large-model era, demand for compute is obvious; the open question is whether supply costs fall as quickly as people expect.",
          "New optimization algorithms may matter. Some researchers, including Hinton, have long questioned whether SGD-based backpropagation is the right long-term path for intelligence.",
        ],
      },
      img(
        "/images/blog/chatgpt-application-technology-04.jpg",
        "Closing AI technology chart from the original post",
        964,
        1440,
        "A closing figure from the original Zhihu essay."
      ),
    ],
  }),
  withEnglish("agent-extra-code", {
    titleEn: "If the Agent Loop Is Tiny, What Are the Other Tens of Thousands of Lines For?",
    excerptEn:
      "A production-engineering answer from Claude Code and Memex: the core ReAct loop is small, but real agents need tools, permissions, UI, state recovery, observability, and product structure.",
    blocksEn: [
      {
        type: "p",
        html: "After Claude Code's source leaked, I ran cloc and saw roughly 400,000 lines. I then asked Claude Code to look at the code with me. The conclusion was not that the core agent idea is complicated. It was that a production agent product contains a huge amount of surrounding engineering.",
      },
      {
        type: "quote",
        html: "From the source, it is a full product-grade application: 40+ tool implementations with validation, permission models, progress tracking, and error handling; 140+ Ink/React terminal UI components; IDE bridge layers for VS Code and JetBrains; OAuth, JWT, macOS Keychain integration, organization policy controls; multi-agent coordination; plugin, skill, and memory systems; slash commands; session recovery; remote mode; voice input; Vim mode; themes; feature flags; telemetry; analytics; and a lot of TypeScript type definitions.",
      },
      img(
        "/images/blog/agent-extra-code-01.jpg",
        "Online discussion about Claude Code source size",
        1440,
        890,
        "A screenshot from the original answer showing people reacting to the code size."
      ),
      {
        type: "p",
        html: "This probably explains why Claude Code can say it was built with Claude, while Anthropic is still hiring many engineers. AI can help write code, but real product engineering does not disappear.",
      },
      {
        type: "p",
        html: "The core logic of an agent can indeed be a simple ReAct-style loop. But making that loop work in the real world requires a large amount of engineering. Since many answers focused on coding agents, I used our recent open-source personal-life-recording agent, Memex, as the example. Its agent logic runs locally on the phone, and the codebase had already grown to nearly 80,000 lines.",
      },
      img(
        "/images/blog/agent-extra-code-02.jpg",
        "Memex codebase size screenshot",
        1122,
        618,
        "Memex had already grown into a substantial local-agent codebase."
      ),
      {
        type: "h2",
        html: "1. Connecting multiple LLM providers",
      },
      {
        type: "p",
        html: "OpenAI's API format is a de facto standard, but not every model provider or cloud vendor is fully compatible. To support multiple providers, you need wrappers for streaming output, token accounting, error handling, and edge cases.",
      },
      {
        type: "p",
        html: "Because the mobile Dart/Flutter ecosystem lacked a mature foundation for this, we open-sourced <a href=\"https://github.com/memex-lab/dart_agent_core\">dart_agent_core</a> to unify these interfaces. That layer alone reached around 7,000 lines.",
      },
      img(
        "/images/blog/agent-extra-code-03.jpg",
        "dart_agent_core code size screenshot",
        1120,
        316,
        "The LLM provider layer alone contains nontrivial engineering work."
      ),
      {
        type: "h2",
        html: "2. Rebuilding the toolbox on mobile",
      },
      {
        type: "p",
        html: "Coding agents such as Devin can rely on a Linux shell and mature command-line tools. A phone does not have that environment. If an agent needs Grep, Find, or Edit-like capabilities, you have to implement those tool behaviors locally.",
      },
      {
        type: "p",
        html: "Memex also handles heterogeneous personal input: images, voice, and text. The agent needs to turn those fragments into structured markdown for management. We deliberately did not provide web search or generic HTTP tools. Memex is meant to focus on personal records and internal logic, not become a general-purpose OpenDevin-like system.",
      },
      {
        type: "h2",
        html: "3. Knowledge management without simple RAG",
      },
      {
        type: "p",
        html: "Coding agents benefit from code's strong structure. Personal records are much messier, and simple retrieval-augmented generation is often not enough. We wrote substantial code and prompts to make the agent behave like a file manager: classifying, indexing, and organizing local records.",
      },
      {
        type: "p",
        html: "We also set strict read-write granularity limits for the knowledge base, so the agent cannot operate on huge files or directories at once and blow up the context. In some organization tasks, we add adversarial review logic: if the agent proposes a knowledge structure that violates rules, code checks reject it and ask the agent to redo the work.",
      },
      {
        type: "h2",
        html: "4. Permissions and safety boundaries",
      },
      {
        type: "p",
        html: "An agent that can call tools and read memory is risky by default. Every tool call needs its own permission checks. The system also needs memory isolation: which data is visible to the agent, and how the agent verifies that it is not operating outside the intended boundary.",
      },
      {
        type: "h2",
        html: "5. Engineering generative UI",
      },
      {
        type: "p",
        html: "We did not want the agent to output only text. Memex experiments with generative UI: a library of common app templates, plus a fallback path where the agent generates HTML and renders it in a WebView. The routing and rendering logic itself takes real engineering.",
      },
      {
        type: "h2",
        html: "6. Process scheduling and state recovery on mobile",
      },
      {
        type: "p",
        html: "Mobile operating systems manage memory aggressively. The app process can be killed at any time. If an agent task is halfway done, the system has to save progress and recover instantly when the user reopens the app.",
      },
      {
        type: "h2",
        html: "7. Observability for model calls",
      },
      {
        type: "p",
        html: "Agent execution produces many model calls. To make the system transparent and controllable, you need observability: how many calls a task used, how many tokens it spent, how much it cost, and where failures occurred.",
      },
      {
        type: "p",
        html: "You also need error tracking. If the agent enters a loop or produces invalid output, logs and automatic interception prevent wasted API spend and make debugging possible.",
      },
      {
        type: "p",
        html: "The core loop is the ideal. The tens of thousands of lines are reality: the code that lets the ideal survive real devices, real data, real users, and real failure modes.",
      },
      {
        type: "p",
        html: "Memex is still only an early prototype. If you are interested in this direction, the project is open at <a href=\"https://github.com/memex-lab/memex\">github.com/memex-lab/memex</a>.",
      },
    ],
  }),
  withEnglish("diffusion-models-guide", {
    titleEn: "A Practical Guide to Diffusion Models",
    excerptEn:
      "A reader-friendly guide to diffusion models from the perspective of someone entering image generation: VAE, ELBO, forward noising, reverse denoising, DDPM training, and sampling.",
    blocksEn: [
      {
        type: "h2",
        html: "Background",
      },
      {
        type: "p",
        html: "For people who came to algorithms through deep learning, diffusion-model papers can feel unusually uncomfortable at first. Compared with many mainstream deep-learning papers, they use more mathematical tools, and in recent years the work with heavier math has often been less visible outside the research community.",
      },
      {
        type: "p",
        html: "If formulas do not scare you away, I would start with <a href=\"https://arxiv.org/abs/2208.11970\">Understanding Diffusion Models: A Unified Perspective</a>. Among the materials I read, it assumes the least background and rarely forces the reader to search for extra context. Then read <a href=\"https://lilianweng.github.io/posts/2021-07-11-diffusion-models/\">What are Diffusion Models?</a> and <a href=\"https://yang-song.net/blog/2021/score/#the-score-function-score-based-models-and-score-matching\">Generative Modeling by Estimating Gradients of the Data Distribution</a>. If you prefer code first, Hugging Face's <a href=\"https://huggingface.co/blog/annotated-diffusion\">The Annotated Diffusion Model</a> is a good PyTorch entry point.",
      },
      {
        type: "p",
        html: "This guide is not a shortcut. It is a set of notes meant to reduce the initial discomfort and help more people build the right basic concepts.",
      },
      {
        type: "h2",
        html: "The basic problem",
      },
      {
        type: "p",
        html: "Generative modeling starts from a dataset drawn from some underlying distribution. If we can fit that data distribution, we can synthesize new samples by sampling from it. In plain language: the images we observe in the real world can be viewed as data from a distribution p(x). If we learn that distribution well enough, we can generate new images from it.",
      },
      {
        type: "p",
        html: "In reality, p(x) is complex. From my learning experience, VAE is the easiest entry point for understanding diffusion models.",
      },
      {
        type: "h3",
        html: "Allegory of the cave",
      },
      img(
        "/images/blog/diffusion-models-01.png",
        "Allegory of the cave diagram",
        740,
        312
      ),
      {
        type: "p",
        html: "The allegory describes people who can only see two-dimensional shadows on a cave wall, while the shadows are projections of three-dimensional objects outside the cave. The point is that the data we observe may be determined by another distribution in a higher or latent space.",
      },
      img(
        "/images/blog/diffusion-models-02.png",
        "Latent variable diagram",
        526,
        326
      ),
      {
        type: "p",
        html: "More formally, there may be a latent variable z that determines the distribution of the observed data x. Because high-dimensional distributions are hard, z is usually lower-dimensional than x.",
      },
      {
        type: "h3",
        html: "The VAE route",
      },
      {
        type: "p",
        html: "A VAE uses two networks: an encoder that maps observed data x into a latent variable z, and a decoder that maps sampled z back to x. In practice, the model trains two parameter sets, often written as p_theta(x|z) and q_phi(z|x).",
      },
      {
        type: "p",
        html: "The VAE objective follows a likelihood-based route. It maximizes the probability of observed data, using the evidence lower bound, or ELBO, as a tractable lower bound.",
      },
      img(
        "/images/blog/diffusion-models-03.png",
        "VAE ELBO formula",
        386,
        82
      ),
      img(
        "/images/blog/diffusion-models-04.png",
        "VAE objective transformation",
        1180,
        229
      ),
      {
        type: "p",
        html: "The two terms in the objective have intuitive meanings: the decoder should reconstruct the image well, and the encoder distribution should stay close to the assumed prior p(z).",
      },
      {
        type: "p",
        html: "The problem is that ELBO is only a lower bound. A loose lower bound is still a lower bound, but it may not produce good samples. VAE also depends on choices such as Gaussian families, reparameterization, and KL divergence because sampling would otherwise break gradient flow.",
      },
      {
        type: "p",
        html: "In short, VAE makes several assumptions to become trainable, and those assumptions limit its ceiling.",
      },
      {
        type: "h2",
        html: "The core idea of diffusion models",
      },
      {
        type: "p",
        html: "Diffusion models can be seen as a successful attempt to reduce the difficulty of the VAE-style problem. Instead of learning both encoder and decoder at once, diffusion fixes the forward process.",
      },
      {
        type: "p",
        html: "If you repeatedly add Gaussian noise to an image, after enough steps it becomes nearly pure Gaussian noise.",
      },
      img(
        "/images/blog/diffusion-models-05.png",
        "Forward diffusion noise process",
        724,
        128
      ),
      {
        type: "p",
        html: "Compared with VAE, the encoder-like forward process is written by us in advance. It does not need to be learned. The model focuses on learning the reverse process: how to reconstruct data step by step from noise.",
      },
      img(
        "/images/blog/diffusion-models-06.png",
        "Reverse diffusion process",
        678,
        139
      ),
      {
        type: "p",
        html: "Because the forward process is stepwise, the reverse process can also be stepwise. The decoder no longer needs to generate the whole image in one jump. It can restore the image little by little, which makes the problem much easier.",
      },
      {
        type: "h2",
        html: "Understanding it as a deep-learning algorithm",
      },
      {
        type: "ol",
        items: [
          "Forward diffusion: take an original image and turn it into Gaussian noise over T steps, with the noise schedule fixed in advance.",
          "Reverse diffusion: train a neural network to learn the distribution that gradually restores the noise image back toward the original image.",
        ],
      },
      {
        type: "p",
        html: "After a series of derivations, the ELBO for diffusion models can be written in a complicated form, then simplified into a very clean objective.",
      },
      img(
        "/images/blog/diffusion-models-07.png",
        "Diffusion ELBO derivation",
        2244,
        1924
      ),
      img(
        "/images/blog/diffusion-models-08.png",
        "Simplified diffusion objective",
        980,
        482
      ),
      {
        type: "p",
        html: "The simplified view is: at each step, train the model to predict the noise that was added. The model does not directly learn the full data distribution. It learns the noise distribution added during the forward process.",
      },
      img(
        "/images/blog/diffusion-models-09.png",
        "Noise prediction loss",
        568,
        60
      ),
      {
        type: "p",
        html: "Most mainstream implementations use U-Net-like structures. The architecture matters, but this guide focuses on the basic process rather than the network-design details.",
      },
      {
        type: "h3",
        html: "Training",
      },
      img(
        "/images/blog/diffusion-models-10.png",
        "DDPM training pseudocode",
        438,
        215
      ),
      {
        type: "ol",
        items: [
          "Start from x_0, an original image from the training set.",
          "Sample a timestep t. Thanks to the math, training does not need to simulate every step from 0 to t each time.",
          "Sample Gaussian noise epsilon.",
          "Feed x_0, epsilon, and t into the model, then train it to predict the noise.",
        ],
      },
      {
        type: "p",
        html: "The key trick is that x_t can be written directly from x_0 and a sampled noise term using reparameterization. That makes training much more efficient because any t can be sampled directly.",
      },
      {
        type: "h3",
        html: "Sampling",
      },
      img(
        "/images/blog/diffusion-models-11.png",
        "DDPM sampling pseudocode",
        436,
        213
      ),
      {
        type: "p",
        html: "Sampling starts from Gaussian noise and walks backward from T to 0. At each step, the model predicts the noise component and uses it to recover the previous image state. Repeating this process gradually denoises the image.",
      },
      {
        type: "h2",
        html: "Summary",
      },
      {
        type: "p",
        html: "This guide only covers the basic concepts. It does not cover the many later developments or unresolved problems in diffusion models. The field has moved quickly, and there is a large body of work worth exploring.",
      },
      img(
        "/images/blog/diffusion-models-12.png",
        "Diffusion model paper landscape",
        1467,
        488
      ),
      {
        type: "p",
        html: "The goal is to make the first encounter less painful: understand the generative-model problem, use VAE to motivate the latent-variable view, then see diffusion as a fixed noising process plus a learned denoising process.",
      },
      img(
        "/images/blog/diffusion-models-13.png",
        "Diffusion guide reference outline",
        2438,
        1596
      ),
      {
        type: "h2",
        html: "References",
      },
      {
        type: "ul",
        items: [
          "<a href=\"https://arxiv.org/abs/2208.11970\">Understanding Diffusion Models: A Unified Perspective</a>",
          "<a href=\"https://lilianweng.github.io/posts/2021-07-11-diffusion-models/\">What are Diffusion Models?</a>",
          "<a href=\"https://yang-song.net/blog/2021/score/#the-score-function-score-based-models-and-score-matching\">Generative Modeling by Estimating Gradients of the Data Distribution</a>",
          "<a href=\"https://huggingface.co/blog/annotated-diffusion\">The Annotated Diffusion Model</a>",
          "<a href=\"https://arxiv.org/pdf/2006.11239.pdf\">Denoising Diffusion Probabilistic Models</a>",
        ],
      },
    ],
  }),
] as const;
