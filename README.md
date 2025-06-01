# AIfinalproject
## Automatic Chinese Text Summarization

This project implements automatic Chinese text summarization using three methods: (1) TextRank, (2) a modified TextRank with adjusted similarity calculations, and (3) the deep learning MT5 model. A simple GUI is also provided for ease of use.

TextRank works by scoring sentences based on their similarity relationships; the top-scoring sentences are selected as the summary. MT5 is a generative seq2seq model, which can be simply understood as predicting the next words based on previous ones. Overall, the deep learning MT5 approach produces better summaries compared to TextRank.

Among the three methods, TextRank is unsupervised and requires no training data. The MT5 model uses the open-source model mT5_multilingual_XLSum from Hugging Face(https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum), fine-tuned using the NLPCC2017 task3 dataset(http://tcci.ccf.org.cn/conference/2017/taskdata.php).

Below are some example results (original Chinese text is retained for clarity).




Here are some sample results

#### Example 1：

7月11日，连续强降雨，让四川登上了中央气象台“头条”，涪江绵阳段水位迅速上涨，洪水一度漫过了宝成铁路涪江大桥桥墩基座，超过封锁水位。洪水在即，中国铁路成都局集团公司紧急调集两列重载货物列车，一前一后开上涪江大桥，每一列货车重量约四千吨，用“重车压梁”的方式，增强桥梁自重，抵御汹涌的洪水。从11日凌晨开始，四川境内成都、绵阳、广元等地连续强降雨，而四川北向出川大动脉—宝成铁路，便主要途径成绵广这一区域。连续的强降雨天气下，绵阳市境内的涪江水位迅速上涨，一度危及到了宝成铁路涪江大桥的安全，上午10时，水位已经超过了涪江大桥上、下行大桥的封锁水位。记者从中国铁路成都局集团公司绵阳工务段了解到，上行线涪江大桥，全长393米，建成于1953年；下行线涪江大桥，全长438米，建成于1995年。“涪江大桥上游有一个水电站，由于洪水太大，水电站已无法发挥调节水位的作用。”情况紧急，铁路部门决定采用“重车压梁”的方式，增强桥梁自重，提高洪峰对桥墩冲刷时的梁体稳定性。简单来说，就是将重量大的货物列车开上涪江大桥，用货车的自重，帮助桥梁抵御汹涌的洪水。恰好，绵阳工务段近期正在进行线路大修，铁路专用的卸砟车，正好停放在绵阳附近。迎着汹涌的洪水，两列重载货车驶向宝成铁路涪江大桥。上午10时30分，第一列46052次货车，从绵阳北站开出进入上行涪江桥。上午11时15分，第二列22001次货车，从皂角铺站进入下行涪江桥。这是两列超过45节编组的重载货物列车，业内称铁路专用卸砟车，俗称“老K车”，车厢里装载的均为铁路道砟，每辆车的砟石的重量在70吨左右。记者从绵阳工务段了解到，货车里满载的砟石、加上一列货车的自重，两列“压桥”的货运列车，每一列的重量超过四千吨。“采用重车压梁的方式来应对水害，在平时的抢险中很少用到。”据了解，在绵阳工务段，上一次采用重车压梁，至少已经是二十年前的事。下午4时许，经铁路部门观测，洪峰过后，涪江水位开始下降，目前已经低于桥梁封锁水位。从下午4点37分开始，两列火车开始撤离涪江大桥。在桥上停留约6个小时后，两列重载货物列车成功完成了“保桥任务”，宝成铁路涪江大桥平安了！（https://m.yybnet.net/chengdu/news/201807/7713943.html）



TextRank:

洪水在即，中国铁路成都局集团公司紧急调集两列重载货物列车，一前一后开上涪江大桥，每一列货车重量约四千吨，用“重车压梁”的方式，增强桥梁自重，抵御汹涌的洪水。
连续的强降雨天气下，绵阳市境内的涪江水位迅速上涨，一度危及到了宝成铁路涪江大桥的安全，上午10时，水位已经超过了涪江大桥上、下行大桥的封锁水位。
简单来说，就是将重量大的货物列车开上涪江大桥，用货车的自重，帮助桥梁抵御汹涌的洪水。



TextRank+TF-IDF:

7月11日，连续强降雨，让四川登上了中央气象台“头条”，涪江绵阳段水位迅速上涨，洪水一度漫过了宝成铁路涪江大桥桥墩基座，超过封锁水位。
洪水在即，中国铁路成都局集团公司紧急调集两列重载货物列车，一前一后开上涪江大桥，每一列货车重量约四千吨，用“重车压梁”的方式，增强桥梁自重，抵御汹涌的洪水。
简单来说，就是将重量大的货物列车开上涪江大桥，用货车的自重，帮助桥梁抵御汹涌的洪水。



mt5:

7月11日,连续强降雨,让四川登上了中央气象台“头条”。四川北向出川大动脉—宝成铁路,也主要途径成绵广。



#### Example 2：

网红李炮儿是一位拥有千万粉丝的网红，喜欢拍摄一些挑战类视频，在这些挑战中，李炮儿凭借着创意的选题，幽默的风格，被大家喜欢。比如挑战过体验印度式洗脸、跑马拉松、当一天特警、cos瑶参加漫展、扮演蜘蛛侠进幼儿园，既搞笑又特别。而要说最让大家喜欢两次挑战，一次是去挑战参加《中国好声音》，网友们本以为是炮灰，哪成想他如此有实力，挺进了决赛，还险些出道。而另一次则是，李炮儿应粉丝要求，去看周杰伦演唱会并且和周杰伦合唱，李炮儿在演唱会上为了吸引周杰伦的注意也是花费了不少心思，终于获得周杰伦的注意，并成功获得了合唱的机会。而近日，李炮儿再次完成了一项挑战，带300多位舞者齐跳科目三，打破了世界纪录，但没想到却遭到全网嘲讽。2024年前夕，李炮儿宣布要做一件大事 ，来为2023年顺利收尾，宣布要挑战一项吉尼斯世界纪录，经过团队的精心选题，最终宣布要挑战带领最多人跳“科目三”。于是，李炮儿联系了吉尼斯认证官，并火速组织了一个几百人的团队（包括线下、线上），租了一个场地加紧训练，还请了专业的舞蹈老师指导。经过了大概一天的训练，李炮儿带领百人团队在辽宁沈阳步行街开始了挑战，百人的团队小伙伴们分成了横竖数排，以此排开。李炮儿则站在最前端领队。当晚正值跨年夜，街上的人十分多，纷纷站在两侧观看。而随着吉尼斯认证官吴晓红，在查验好人数和队形，在台上宣布“世界上最多人跳科目三”项目挑战开始后，音乐响起，数百人团队开始一起随音乐扭动，场面十分之“壮观”，而于此同时，线上还有200多人同时视频连线挑战。不过，在 第一次挑战过程中，线上的网络出了一些问题，导致线上的小伙伴出现了掉线、卡顿等状况，所以，第一次挑战失败。于是在重新检查网络后，开始了第二次挑战。而在第二次挑战前，李炮儿也是为现场和线上的小伙伴们打气，鼓励大家加油，一定要挑战成功。最终，在大家的共同努力下，第二次挑战顺利完成，而吉尼斯认证官在统计线上、线下的舞者人数后，高兴地在台上宣布最终完成人数为357人，宣布李炮儿打破了一项 新的吉尼斯世界纪录。而李炮儿也上台领取了吉尼斯认证奖牌，和小伙伴来了一张大合影，还上台表达了得奖的感言，并为家乡宣传，希望大家来沈阳游玩。不过，李炮儿在发布视频后，没有像以往一样迎来大家的夸赞，而是引来一片嘲讽声音。翻看网友们的嘲讽原因，大致有以下三个方面：第一， 感觉这次挑战的内容没啥难度，也没有什么特点，只要人足够多，就肯定能挑战成功。而且挑战成功的条件十分地简单，只要所有的参与者知道动作，跳够两分钟就算成功。简直就是小朋友过家家。那么按照李炮儿的这种挑战，下一次阿giao可以找些粉丝挑战最多人在一起“giao”的记录了，面筋哥可以组织一波最多人一起唱“烤面筋”的吉尼斯世界纪录。第二， 网友们也感觉“吉尼斯”个项目越来越尴尬，不少网友直言，以前感觉吉尼斯纪录都是好厉害的样子，而现在感觉吉尼斯记录就是个笑话。实际上，吉尼斯世界纪录并非所有的记录都十分有难度，而且很多记录都十分奇葩，比如有用屁股坐核桃、大腿夹西瓜、甚至各种因为太无聊，没有人继续挑战的，在这些记录中，以堆人数而挑战成功的也有很多。比如四川曾有2.2万大妈一起跳广场舞打破吉尼斯世界纪录。只不过，李炮儿前期的视频要么给大家带来快乐，要么是实打实的有实力，而这次虽然花费了很大的力气，也找了很多人，但确实是毫无挑战性，和大家的预期相差甚远。当然，李炮儿利用这次活动，为家乡宣传，初心还是十分好的。（https://www.sohu.com/a/749023640_121736089）



TextRank:

2024年前夕，李炮儿宣布要做一件大事，来为2023年顺利收尾，宣布要挑战一项吉尼斯世界纪录，经过团队的精心选题，最终宣布要挑战带领最多人跳“科目三”。
最终，在大家的共同努力下，第二次挑战顺利完成，而吉尼斯认证官在统计线上、线下的舞者人数后，高兴地在台上宣布最终完成人数为357人，宣布李炮儿打破了一项新的吉尼斯世界纪录。
那么按照李炮儿的这种挑战，下一次阿giao可以找些粉丝挑战最多人在一起“giao”的记录了，面筋哥可以组织一波最多人一起唱“烤面筋”的吉尼斯世界纪录。



TextRank+TF-IDF:

而在第二次挑战前，李炮儿也是为现场和线上的小伙伴们打气，鼓励大家加油，一定要挑战成功。
最终，在大家的共同努力下，第二次挑战顺利完成，而吉尼斯认证官在统计线上、线下的舞者人数后，高兴地在台上宣布最终完成人数为357人，宣布李炮儿打破了一项新的吉尼斯世界纪录。
那么按照李炮儿的这种挑战，下一次阿giao可以找些粉丝挑战最多人在一起“giao”的记录了，面筋哥可以组织一波最多人一起唱“烤面筋”的吉尼斯世界纪录。



mt5:

近日,网红李炮儿再次完成一项挑战,带300多位舞者齐跳科目三,打破了世界纪录,但没想到却遭到全网嘲讽。





## How to Use

First, install the required dependencies:

```
python >= 3.9
jieba
numpy
scikit-learn
transformers
sentencepiece
pytorch
```
If you have questions about installing PyTorch, refer to this link(https://pytorch.org/get-started/previous-versions/).

To run the program, simply execute:
python text_generate.py

This will launch the GUI.

Note: Due to the large size of the MT5 model, it has been removed from this repository. This does not affect TextRank usage. You can download the MT5 model folder here: https://drive.google.com/drive/folders/1eB2F_ZF7tDInGgHtp76ZGrBsgoe4MxbG?usp=sharing

After downloading, place the mt5-base folder in the same directory as text_generate.py, then rerun the program. Please note that MT5 generation can be slow, so be patient during processing.






## Methods

### TextRank

TextRank builds an undirected graph , where each sentence is a node , and edges represent the similarity between sentences. The initial score of each node is set to 1, and scores are updated using the formula:


$$
WS(N_i) = (1 - d) + d \times \sum_j \frac{w_{ji}}{\sum_k w_{jk}} \times WS(N_j)
$$


where  is the damping factor (typically 0.85). The edge weights  define sentence similarity, which can be measured in various ways.

#### Similarity Measures

##### Original Method 


Original Method ([from TextRank paper](https://aclanthology.org/W04-3252.pdf)): Counts the number of shared words between two sentences, normalized by sentence lengths.



TF-IDF Based: Converts each sentence into a TF-IDF vector and calculates cosine similarity between vectors.






### MT5

MT5 is Google’s multilingual text-to-text Transformer model. It is pretrained on large-scale data and fine-tuned for summarization tasks. For more details, see the paper[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://research.google/pubs/exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer/)。



## Discussion
Although TextRank is an early work, it already exhibits the essence of graph-based reasoning. Its effectiveness comes from combining local sentence context with global graph-level recursive information. However, it has limitations:

It selects sentences directly from the text, which may miss the true main ideas if no single sentence represents them.

It treats each word independently, ignoring synonymy and co-reference relationships.

Integrating word embeddings (e.g., Word2Vec) or transformer-based similarity measures could improve the system’s performance.


## The `data/` folder contains:
- example_input.txt → original input texts
- generated_output.txt → summaries produced by each method
- rouge_scores.txt → ROUGE score evaluation results




## License
This project is released under the MIT License.


## Acknowledgements
- Hugging Face Transformers Library
- "TextRank: Bringing Order into Texts" by Rada Mihalcea and Paul Tarau
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 paper)
- NLPCC 2017 Task 3 dataset


