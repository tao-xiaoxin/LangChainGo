## 一、大语言模型简介
大语言模型是一种人工智能模型，通常使用深度学习技术，来理解和生成人类语言。这些模型的“大”在于它们的参数数量非常多，可以达到数十亿甚至更多,通常模型的**1B代表1亿参数量**，这使得它们能够理解和生成高度复杂的语言模式。

大语言模型可以想象成一个巨大的预测机器，其训练过程主要基于“猜词”：给定一段文本的开头，它的任务就是预测下一个词是什么。模型会根据大量的训练数据（例如在互联网上爬取的文本），试图理解词语和词组在语言中的用法和含义，以及它们如何组合形成意义。它会通过不断地学习和调整参数，使得自己的预测越来越准确。

比如我们给模型一个句子：“今天的天气真”，模型可能会预测出“好”作为下一个词，因为在它看过的大量训练数据中，“今天的天气真好”是一个常见的句子。这种预测并不只基于词语的统计关系，还包括对上下文的理解，甚至有时能体现出对世界常识的认知，比如它会理解到，人们通常会在天气好的时候进行户外活动。因此也就能够继续生成或者说推理出相关的内容。

大语言模型并不完全理解语言，它们没有人类的情感、意识或理解力。**它们只是通过复杂的数学函数学习到的语言模式，一个概率模型来做预测**，所以有时候它们会犯错误，或者生成不合理甚至偏离主题的内容。

![模型预测](http://billy.taoxiaoxin.club/md/2025/03/546630a6735808692557e8152ac94959.png)

## 二、LangChain 基本介绍

### 2.1 基本概念
LangChain 是一个全方位的、基于大语言模型这种预测能力的应用开发框架，它的灵活性和模块化特性使得处理语言模型变得极其简便。不论你在何时何地，都能利用它流畅地调用语言模型，并基于语言模型的“预测”或者说“推理”能力开发新的应用。所以，它使得应用程序能够：

+ **具有上下文感知能力**：将语言模型连接到上下文来源（提示指令，少量的示例，需要回应的内容等）
+ **具有推理能力**：依赖语言模型进行推理（根据提供的上下文如何回答，采取什么行动等）

![LangChain 的标志，我想是1只能说会道的鹦鹉+1个链条](http://billy.taoxiaoxin.club/md/2025/03/5d9c0afab907e232a7cd910464173b73.png)

### 2.2 架构

LangChain 框架由多个开源库组成。请在[架构](https://python.langchain.ac.cn/docs/concepts/architecture/)页面阅读更多内容。

- **`langchain-core`**：聊天模型和其他组件的基本抽象。
- **集成包**（例如，`langchain-openai`、`langchain-anthropic`等）：重要的集成已被拆分为轻量级软件包，由 LangChain 团队和集成开发人员共同维护。
- **`langchain`**：构成应用程序认知架构的链、代理和检索策略。
- **`langchain-community`**：由社区维护的第三方集成。
- **`langgraph`**：用于将 LangChain 组件组合成可用于生产的应用程序的编排框架，具有持久性、流式传输和其他关键功能。请参阅[LangGraph 文档](https://github.langchain.ac.cn/langgraph/)。

### 2.3 简化了 LLM 应用程序生命周期

LangChain 简化了 LLM 应用程序生命周期的每个阶段

- **开发**：使用 LangChain 的开源[组件](https://python.langchain.ac.cn/docs/concepts/)和[第三方集成](https://python.langchain.ac.cn/docs/integrations/providers/)构建您的应用程序。使用[LangGraph](https://python.langchain.ac.cn/docs/concepts/architecture/#langgraph) 构建具有一流流式传输和人工参与支持的状态代理。
- **生产化**：使用[LangSmith](https://docs.smith.langchain.com/)检查、监控和评估您的应用程序，以便您可以持续优化并充满信心地部署。
- **部署**：使用[LangGraph 平台](https://github.langchain.ac.cn/langgraph/cloud/)将您的 LangGraph 应用程序转换为可用于生产的 API 和助手。

![LLM 应用程序生命周期](http://billy.taoxiaoxin.club/md/2025/03/10e8cf6fbdaf092980e01981649555dc.png)


### 2.4 相关文档
+ [官网系列教程](https://python.langchain.com/docs/introduction/)
+ [GitHub 开源地址](https://github.com/langchain-ai/langchain)



### 2.5 LangChain安装


```python
# 2.5.1 LangChain 的基本安装非常简单，只需使用以下命令：
!pip install langchain

# 2.5.2 升级到最新版
!pip install --upgrade langchain

# 2.5.3 安装 LangChain 并包括常用的开源 LLM（大语言模型）库：
!pip install langchain[llms]

# 2.5.4 从源代码安装
!git clone https://github.com/langchain-ai/langchain.git
!cd langchain
!pip install -e .
```

## 三、搭建模型环境
这里我们使用腾讯云`CloudStudio`提供的开发环境和算力资源，打开地址：[`https://ide.cloud.tencent.com/dashboard/`](https://ide.cloud.tencent.com/dashboard/)

选择使用`QwQ-32B` 模型模板，点击创建即可。

![QwQ-32B模型模版](http://billy.taoxiaoxin.club/md/2025/03/0e6be992588bd4c1b3de46fef1fba2d6.png)

生成并复制ssh 链接。
![ssh链接生成](http://billy.taoxiaoxin.club/md/2025/03/f15ca6c7771ecf2d0e7c3a6e71e4bc8d.png)
![QwQ-32B模型模版](http://billy.taoxiaoxin.club/md/2025/03/9a4516b1d31d95efce986ca9f05b4817.png)


接着先验证下能否ssh，从上边复制出来的：
```bash
ssh -p 22 ******@sllfys.clvol20v.ssh.cloudstudio.work
```

查看下腾讯云主机`ollama`的运行状态：
```bash
➜  ~ ollama list
NAME                ID              SIZE      MODIFIED
bge-m3:latest       790764642607    1.2 GB    3 days ago
deepseek-r1:1.5b    a42b25d8c10a    1.1 GB    3 days ago
deepseek-r1:7b      0a8c26691023    4.7 GB    3 days ago
deepseek-r1:8b      28f8fd6cdc67    4.9 GB    3 days ago
```

找到ollama的服务端口，也就是8434
```bash
➜  ~ netstat -tpln|grep ollama
tcp6       0      0 :::8434                 :::*                    LISTEN      2141/ollama
```

通过ssh创建本地端口映射
```bash
ssh -p 22 -f -N -L 8434:127.0.0.1:8434 ******@sllfys.clvol20v.ssh.cloudstudio.work
```

> 具体可以参考腾讯云文档:[https://cloud.tencent.com/document/product/1039/116081](https://cloud.tencent.com/document/product/1039/116081)

验证下本机服务是否通过ssh隧道打通
```bash
curl  -s http://127.0.0.1:8434/v1/models
```
响应如下：
```bash
{
    "object": "list",
    "data": [
        {
            "id": "qwq:32b",
            "object": "model",
            "created": 1741229839,
            "owned_by": "library"
        }
    ]
}
```

## 四、OpenAI API 的基本使用
LangChain 本质上就是对各种大模型提供的 API 的套壳，是为了方便我们使用这些 API，搭建起来的一些框架、模块和接口。所以，我们要对OpenAI 的 API 有进一步的了解。

OpenAi API文档:[https://platform.openai.com/docs/guides/text](https://platform.openai.com/docs/guides/text)

### 4.1 OpenAi 库的安装


```python
!pip install openai
```

### 4.2 基本使用


```python
from openai import OpenAI
from config import config

# Get configuration from environment variables
api_key = config.OPENAI_API_KEY
api_base = config.OPENAI_API_BASE
model_name = config.OPENAI_MODEL_NAME

client = OpenAI(api_key=api_key, base_url=api_base)
response = client.completions.create(
    model=model_name,
    temperature=0.6,
    prompt="Deepseek 英文单词有几个字母e?",
    max_tokens=500,
)
```

### 4.3 Chat 方法 vs Text 方法

Chat 模型和 Text 模型都有各自的优点，其适用性取决于具体的应用场景。相较于 Text 模型，Chat 模型的设计更适合处理对话或者多轮次交互的情况。这是因为它可以接受一个消息列表作为输入，而不仅仅是一个字符串。这个消息列表可以包含 `system`、`user `和 `assistant` 的历史信息，从而在处理交互式对话时提供更多的上下文信息。
这种设计的主要优点包括：

+ 对话历史的管理：通过使用 Chat 模型，你可以更方便地管理对话的历史，并在需要时向模型提供这些历史信息。例如，你可以将过去的用户输入和模型的回复都包含在消息列表中，这样模型在生成新的回复时就可以考虑到这些历史信息。
+ 角色模拟：通过 `system` 角色，你可以设定对话的背景，给模型提供额外的指导信息，从而更好地控制输出的结果。

当然在 Text 模型中，你在提示中也可以为 AI 设定角色，作为输入的一部分。然而，对于简单的单轮文本生成任务，使用 Text 模型可能会更简单、更直接。例如，如果你只需要模型根据一个简单的提示生成一段文本，那么 Text 模型可能更适合。从上面的结果看，Chat 模型给我们输出的文本更完善，是一句完整的话，而 Text 模型输出的是几个名字。这是因为现在的大模型基于人类反馈的强化学习，输出的答案更像是真实聊天场景。


#### 4.3.1 Text 方法介绍

Text 方法（也称为 Completions API）是 OpenAI 提供的较早的 API 接口，主要用于文本生成任务。它接收一个文本提示（prompt），然后模型基于此提示继续生成文本。

##### (一)、Text 请求常见参数介绍
| 序号 | 参数              | 含义                                                         |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | `model`           | 模型的类型，这里使用的是阿里通义千问最新出的`QvQ-32B`模型。    |
| 2    | `prompt`          | 提示，也就是输入给模型的问题或者指示，告诉模型我们要它做什么。 |
| 3    | `temperature`     | 这个参数可以影响模型输出的随机性。值越高（接近1），输出就越随机；值越低（接近0），输出就越确定。例如，如果你设置 `temperature=0.8`，模型会有较高的可能性生成不同的输出；如果你设置 `temperature=0.2`，模型生成的输出就会更加一致，更有可能重复相同的输出。 |
| 4    | `max_tokens`      | 这个参数可以限制模型输出的最大长度，注意这个长度是以 `Tokens` 为单位的，而一个 `Token` 可以是一个字、一个词或一个字符，这取决于语言和模型。 |
| 5    | `suffix`          | 这个参数允许用户为模型生成的输出文本后附加一个后缀。例如，如果你想每次模型生成的文本都以某种特定格式或标记结尾，你可以使用这个参数。默认情况下，它设置为 `null`，意味着没有后缀将被添加。如果指定了一个字符串作为此参数的值，那么该字符串将被追加到每个输出文本的末尾。 |
| 6    | `top_p`           | 这是与 `temperature` 参数类似的另一个参数，它使用所谓的核心抽样。模型将只考虑概率质量最高的 `Tokens`。例如，`top_p` 设置为 `0.1` 意味着只有前 10% 的最有可能的 `Tokens` 会被考虑。 |
| 7    | `n`               | 这个参数决定了为每个提示生成多少个完整的输出。需要注意的是，使用这个参数可能会更快地耗尽你的 `Token` 配额。 |
| 8    | `stream`          | 这个参数决定是否实时流式传输生成的 `Tokens`。如果设置为 `true`，`Tokens` 将在生成时被发送。 |
| 9    | `logprobs`        | 这个参数要求 API 包含最有可能的 `Tokens` 的对数概率。例如，如果 `logprobs` 设置为 `5`，API 将返回最有可能的 5 个 `Tokens` 的列表。 |
| 10   | `echo`            | 如果设置为 `true`，除了生成的完成内容外，还会回显提示。        |
| 11   | `stop`            | 这个参数允许你指定一个或多个序列，当模型遇到这些序列时，它会停止生成 `Tokens`，返回的文本将不包含 `stop` 序列。 |
| 12   | `presence_penalty`| 这是一个在 `-2.0` 到 `2.0` 之间的数字。正值会惩罚已经出现在文本中的新 `Tokens`，这可以帮助模型更多地谈论新话题。 |
| 13   | `frequency_penalty` | 这也是一个在 `-2.0` 到 `2.0` 之间的数字。正值会惩罚到目前为止在文本中频繁出现的 `Tokens`，这可以减少模型重复相同内容的可能性。 |
| 14   | `best_of`         | 这个参数会在服务器端生成 `best_of` 个完整的输出，并返回其中最好的一个（即每个 `Token` 的对数概率最高的那个）。与 `n` 参数结合使用时，`best_of` 决定了候选完成的数量，而 `n` 决定了返回的数量。 |
| 15   | `logit_bias`      | 这个参数可以修改指定 `Tokens` 在完成中出现的可能性。接受一个 JSON 对象，该对象将 `Tokens` 映射到 `-100` 到 `100` 之间的偏置值。 |
| 16   | `user`            | 这是一个可选参数，表示你的最终用户的唯一标识符，可以帮助 OpenAI 监测和检测滥用。 |

##### (二)、Text 请求常见响应介绍
打印输出大模型返回的文字：



```python
print(response.choices[0].text.strip())
```

API 响应对象的主要字段包括：
序号 | 字段 | 含义
--- | --- | ---
1 | id | 响应的唯一标识符。
2 | object | 表示该响应的对象类型，对于生成操作，这个字段通常为 text_completion。
3 | created | 表示响应生成的时间。
4 | model | 表示生成响应的模型的名称。
5 | choices | 一个列表，其中包含了模型生成的所有输出。除非指定 n 参数，否则通常只包含一个条目，即索引 [0]。
6 | usage | 提供了关于文本生成过程的统计信息，包括 prompt_tokens（提示的 Token 数量）、completion_tokens（生成的 Token 数量）、total_tokens（总的 Token 数量）。

#### 4.3.2 Chat 方法介绍

Chat 方法（也称为 Chat Completions API）是 OpenAI 更新的 API 接口，专为对话场景设计，支持多轮对话和不同角色（用户、助手、系统）的消息。通常情况下，我们都会通过Chat 接口构建自己的Ai 应用系统。

Chat 请求示例如下：


```python
from openai import OpenAI
from config import config

# Get configuration from environment variables
api_key = config.OPENAI_API_KEY
api_base = config.OPENAI_API_BASE
model_name = config.OPENAI_MODEL_NAME

client = OpenAI(api_key=api_key, base_url=api_base)

response = client.chat.completions.create(  
  model=model_name,
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请问英文单词草莓有几个r字母？"},
    ],
  temperature=0.6,
)
```

我们打印Chat 响应：

##### (一)、Chat 请求常见响应介绍
Chat 模型生成内容后，返回的响应内容字段如下：

| 序号 | 字段   | 含义                                                                 |
|----|------|----------------------------------------------------------------------|
| 1  | id   | 是 OpenAI 给这个生成任务分配的唯一标识符。                                   |
| 2  | object | 是对象类型的标志，在这个例子中，它标志着这是一个聊天模型的生成任务。                   |
| 3  | created | 是生成任务创建的时戳。                                                      |
| 4  | model | 是用来生成任务的模型名称。                                                     |
| 5  | choices | 是一个列表，包含了模型生成的所有条目。<br>除非指定 n 参数，否则通常只包含一个条目，即索引 [0]。<br>条目内部又包含了：<br>· message：包含两个字段，分别是 role （消息的角色，可以是 system、user 或 assistant）和 content （消息的内容）。<br>· finish_reason：表示生成任务结束的原因。它可以 是 stop （遇到停止符）、length （达到了最大长度）或 content_filter （被 OpenAI 的内容过滤器移除）。<br>· index：表示这个选项在 choices 列表中的索引位置。 |
| 6  | usage | 提供了关于文本生成过程的统计信息，包括 prompt_tokens （提示的 Token 数量）、completion_tokens （生成的 Token 数量）、total_tokens （总的 Token 数量）。 |


## 五、LangChain 调用 Text 和 Chat 接口

### 5.1 调用Text 接口



```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(config.OPENAI_MODEL_NAME, model_provider="ollama")
resp = llm.invoke("请问英文单词草莓有几个r字母？")
print(resp.content)
```

### 5.2 调用Chat 接口


```python
from config import config
from langchain.chat_models import init_chat_model
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

llm = init_chat_model(config.OPENAI_MODEL_NAME, model_provider="ollama")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Deepseek 英文单词有几个e？"),
]

resp = llm.invoke(messages)
print(resp.content)

```

## 六、小结
LangChain 当你需要对接很多模型的时候，不同的模型调用方式是不一样的，就会出现很多局限性，LangChain 就是在这些模型上做了一层抽象作为这些模型的`Base model` 基类，然后派生出很多模型的子类，每个子类内置了很多对应模型模型的工具和方法，我们只需要通过配置对应提示词模版输入输出，在它提供的对应工具上开发对应模型的调用方法即可。

总的来说，LangChain通过创建统一的抽象接口来解决不同模型API之间的差异问题。这种设计模式类似于面向对象编程中的"基类-子类"继承关系。

主要优点包括：

1. 统一接口：通过基类抽象，开发者可以用相同的方式与不同的模型交互
2. 减少重复代码：不需要为每个模型重写相似的代码
3. 提高可维护性：更换底层模型时，上层应用代码无需大量修改
4. 工具集成：LangChain为各种模型提供了丰富的工具和方法

这种设计让开发者能专注于业务逻辑，而不是处理模型间的差异。通过配置提示词模板和使用LangChain提供的工具，可以快速开发出适用于不同模型的应用。

不过值得补充的是，LangChain不仅仅提供了模型抽象，它还提供了完整的应用开发框架，包括记忆系统、链式调用、代理系统等高级功能，这些都建立在模型抽象的基础上。

总的来说，你的理解是正确的，LangChain确实通过抽象层解决了多模型接入的问题。
