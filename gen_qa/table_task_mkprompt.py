# Define the tasks
# 1. 表格数据检索
# 2. 数值运算：计数，排序，区间，过滤，计算(加减乘除)，复杂运算，
# 3. 判断题: 
# 4. 选择题: 
# 5. Summary or caption: FreeTQA, summary 


#==========================================Calculations_example
#平均值
markdown_Calculations_exmaple1 = """
Table:
| Day       | Wind speed (miles per hour) |
|-----------|-----------------------------|
| Thursday  | 50 |
| Friday    | 50 |
| Saturday  | 47 |
| Sunday    | 49 |
| Monday    | 49 |
| Tuesday   | 48 |
| Wednesday | 50 |

QUESTION: Lester tracked the maximum daily wind speed for 7 days. What is the mean of the numbers?
COT:Read the numbers from the table.\n\n50, 50, 47, 49, 49, 48, 50\n\nFirst, count how many numbers are in the group.\n\nThere are 7 numbers.\n\nNow add all the numbers together:\n\n50 + 50 + 47 + 49 + 49 + 48 + 50 = 343\n\nNow divide the sum by the number of numbers:\n\n343 / 7 = 49\n\nThe mean is 49.
ANSWER: 49.
"""

#中位数，众数
markdown_Calculations_exmaple2 = """
Table:
| Name    | Score |
|---------|-------|
| Tina    | 10    |
| Samuel  | 6     |
| Lorenzo | 3     |
| Erin    | 6     |
| Hanson  | 6     |
| Grayson | 9     |
| Fred    | 10    |
| Janice  | 6     |

QUESTION: What is median score of this table?
DETAIL_ANSWER: The median score is the middle value when all scores are arranged in ascending order. In this case, when arranged: 3, 6, 6, 6, 6, 9, 10, 10, the middle value is the average of the two middle scores, which is (6+6)/2 = 6.
SHORT_ANSWER: 6

QUESTION: Some friends compared the score. What is the mode of the score?
DETAIL_ANSWER: Read the Score column from the table.\n\n10,6,3,6,6,9,10,6.\n\nFirst, arrange the numbers from smallest to largest:\n\n3,6,6,6,6,9,10,10.\n\nNow count how many times each number appears.\n\n3 appears 1 time.\n6 appears 4 times, 9 appears 1 time, 10 appears 2 times.\n\nThe number that appears most often is 6.\n\nThe mode is 16.
SHORT_ANSWER: 6.
"""

#减法
markdown_Calculations_exmaple3 = """
Table:
|Month|Number of magazines|
|-|-|
|November|3,874|
|December|1,300|
|January|7,828|
|February|5,211|

QUESTION: A magazine editor looked at her sales figures for the past 4 months. How many more copies were sold in February than in November?
DETAIL_ANSWER: Find the numbers in the table.\n\nFebruary: 5,211\nNovember: 3,874\n\nNow subtract: 5,211 - 3,874 = 1,337.\n\n1,337 more copies were sold in February.
SHORT_ANSWER: 1,337.
"""

#加法
markdown_Calculations_exmaple4 = """
Table:
|Month|Number of magazines|
|-|-|
|November|3,874|
|December|1,300|
|January|7,828|
|February|5,211|

QUESTION: What is the total number of magazines for November and December?
DETAIL_ANSWER: 3,874 (November) + 1,300 (December) = 5,174. So the total number of magazines for November and December is 5,174.
SHORT_ANSWER: 5,174.

QUESTION: What is the total number of magazines for November, December, January, and February?
DETAIL_ANSWER: 3,874 (November) + 1,300 (December) + 7,828 (January) + 5,211 (February) = 18,213. So the total number is 18,213.
SHORT_ANSWER: 18,213.
"""

#除法
markdown_Calculations_exmaple4 = """
Table:
|Month|Number of magazines|
|-|-|
|November|3,874|
|December|1,300|
|January|7,828|
|February|5,211|

QUESTION: What is the magazine growth rate from November to December?
DETAIL_ANSWER: The magazine growth rate from November to December is equal to (1300 - 3874) / 3874  * 100, approximately -66.44.
SHORT_ANSWER: -66.44.

QUESTION: What percentage of the total number of magazines in all months is represented by the number of magazines in January?
DETAIL_ANSWER: To find what percentage of the total number of magazines in all months is represented by the number of magazines in January, follow these steps: \n\n1. Calculate the total number of magazines: Total = 3,874 + 1,300 + 7,828 + 5,211 = 18,213\n\n2. Find the number of magazines in January\n\nJanuary = 7,828\n\n3. Calculate the percentage: Percentage = (7,828 / 18,213) × 100 ≈ 43.0%\nThus, the percentage is 43.0.
SHORT_ANSWER: 43.0.    
"""

#乘法
markdown_Calculations_exmaple5 = """
Table:
| peanut butter cup | gummy bear | piece of mint chocolate | piece of licorice | piece of gum |
|--------------------|------------|-------------------------|-------------------|--------------|
| $0.19              | $0.10      |  $0.14                  | $0.19             |   $0.13      |

QUESTION: How much money does Sidney need to buy a piece of mint chocolate and 7 peanut butter cups?
DETAIL_ANSWER: Find the cost of 7 peanut butter cups.\n\n$0.19 × 7 = $1.33\n\nNow find the total cost.\n\n$0.14 + $1.33 = $1.47\n\nThus, Sidney needs $1.47.
SHORT_ANSWER: $1.47.
"""

markdown_Calculations_exmaples = [markdown_Calculations_exmaple1, markdown_Calculations_exmaple2, markdown_Calculations_exmaple3, markdown_Calculations_exmaple4, markdown_Calculations_exmaple5]


#==========================================markdown_Composition_example
markdown_Composition_example1 = """
Table:
| Name    | Score |
|---------|-------|
| Tina    | 10    |
| Samuel  | 6     |
| Lorenzo | 3     |
| Erin    | 6     |
| Hanson  | 6     |
| Grayson | 9     |
| Fred    | 10    |
| Janice  | 6     |

QUESTION: What is the average score of those individuals with scores lower than the second-largest score?
DETAIL_ANSWER: To find the average score of individuals with scores lower than the second-largest score in the table you provided, let's first identify the second-largest score:\nScores listed: 10, 6, 3, 6, 6, 9, 10, 6.\nUnique Scores when sorted: 3, 6, 9, 10.\nThe second-largest score is 9.\nNext, we consider the individuals with scores lower than 9, which are: Samuel: 6\n Lorenzo: 3\n Erin: 6\n Hanson: 6\n Janice: 6\nThese scores are 6, 3, 6, 6, and 6. Now, let's calculate the average of these scores:\nAverage = (6+3+6+6+6)/5 = 27/5 = 5.4\nTherefore, the average score of those individuals with scores lower than the second-largest score is 5.4.
SHORT_ANSWER: 5.4.
"""
markdown_Composition_examples = [markdown_Composition_example1]


#==================================================markdown_Multiple_choice_examples

markdown_Multiple_choice_example1 = """
Table:
| ORBITAL EVENT      | PERIOD OF DAYLIGHT | PERIOD OF NIGHT  |
|--------------------|--------------------|------------------|
| The summer solstice is the day with | longest period of daylight and the   | shortest period of night  |
| The winter solstice is the day with | shortest period of daylight and the  | longest period of night   |
| The spring equinox is the day with  | midrange period of daylight and the  | midrange period of night  |
| The fall equinox is the day with    | midrange period of daylight and the  | midrange period of night  |

QUESTION: Choose the correct answer option based on the table and the question.\nThe spring equinox is the day with the _______ period of daylight?\n1. midrange\n2. shortest\n3. longest\n4. None of the above.
DETAIL_ANSWER: Based on the provided table, the correct answer option: 1. midrange.
SHORT_ANSWER: 1. midrange.
"""

markdown_Multiple_choice_example2 = """
Table:
| ORBITAL EVENT      | PERIOD OF DAYLIGHT | PERIOD OF NIGHT  |
|--------------------|--------------------|------------------|
| The summer solstice is the day with | longest period of daylight and the   | shortest period of night  |
| The winter solstice is the day with | shortest period of daylight and the  | longest period of night   |
| The spring equinox is the day with  | midrange period of daylight and the  | midrange period of night  |
| The fall equinox is the day with    | midrange period of daylight and the  | midrange period of night  |

QUESTION: Identify the correct option by referring to the table and the problem statement. Question:\nThe winter solstice has a _______ period of night compared to the other orbital events.\nOptions:\n(A) tiny\n(B) short\n(C) midrange\n(D) long.
DETAIL_ANSWER: Based on the table information, the correct answer is (D) long.
SHORT_ANSWER: (D) long.
"""

markdown_Multiple_choice_example3 = """
Table:
| (in millions)Fiscal Year | AmortizationExpense |
|--------------------------|---------------------|
| 2017                     | $1,931              |
| 2018                     | $1,898              |
| 2019                     | $1,805              |
| 2020                     | $1,757              |
| 2021                     | $1,739              |

QUESTION: Based on the table, in which year was the company's Amortization Expense the highest?\nOptions:\n(A) 2018\n(B) 2019\n(C) 2020\n(D) 2017.
DETAIL_ANSWER: According to the table, the Amortization Expense in 2017 was $1,931 million, which is the highest among all the years listed. So, the correct answer is (D) 2017.
SHORT_ANSWER: (D) 2017.
"""

markdown_Multiple_choice_examples = [markdown_Multiple_choice_example1, markdown_Multiple_choice_example2, markdown_Multiple_choice_example3]


#==================================================markdown_FactVerfied_examples

markdown_FactVerfied_example1 = """
Table:
| language | padilla municipality | tomina municipality |
|----------|----------------------|---------------------|
| quechua  | 2181                 | 7831                |
| aymara    | 29                   |    23               |

QUESTION: Based on the table image and the subsequent sentence, determine if the table entails or disputes the subsequent sentence.\nsentence: the padilla municipality language for quechua is 2181.
DETAIL_ANSWER: Based on the table, the number for the Quechua language in Padilla Municipality is 2181. So, the answer is entails.
SHORT_ANSWER: entails.
"""

markdown_FactVerfied_example2 = """
Table:
| language | padilla municipality | tomina municipality |
|----------|----------------------|---------------------|
| quechua  | 2181                 | 7831                |
| aymara    | 29                  |    23              |

QUESTION: Considering the following sentence 'the padilla municipality language for quechua is 2182.' and using the table as premise, please determine whether the table 'upholds' or 'challenges' the above sentence. If the evidence in the table is not sufficient for a sound judgement, classify the sentence as 'neutral'.
DETAIL_ANSWER: The table states that the number for the Quechua language in Padilla Municipality is 2181, while the sentence claims it is 2182.\nSo, the table challenges the given sentence.
SHORT_ANSWER: challenges.
"""

markdown_FactVerfied_example3 = """
Table:
| week    | date          |  opponent   | result           |   attendance  |
|---------|---------------|-------------|------------------|---------------|
|  14     | december 5, 1988 | chicago bears |  w 23 - 3   | 65579         |
|  15     | december 11, 1988 | atlanta falcons |  w 22 - 7  | 42828       |
|  16     | december 18, 1988 | san francisco 49ers |  w 38 - 16  | 62444  |

QUESTION: Analyze the table's content, and determine whether the table affirms or disputes the given sentence.\nsentence: the total score in week 16 was lower than the total score in week 15.
DETAIL_ANSWER: The table shows that the total score for week 15 is 22 + 7 = 29, while for week 16 it is 38 + 16 = 54.\nThe total score of week 16 is higher. Thus, the table disputes the sentence.
SHORT_ANSWER: disputes.
"""

markdown_FactVerfied_examples = [markdown_FactVerfied_example1, markdown_FactVerfied_example2, markdown_FactVerfied_example3]


#==================================================markdown_Freetqa_examples
markdown_Freetqa_example1 = """
Table:
| Party                        | Votes                     |   %         |  Seats        |
|--------------------------------|---------------------------|-------------|-------------|
| Vietnamese Fatherland Front  | 10,561,314                |  100        |   424         |
| Invalid/blank votes          | 105,807                   |         -   |   -           |
|Total                         | 10,667,121                |  100        |   424         |

QUESTION: What were the voting results of the 1975 North Vietnamese legislative election with regards to seats?
DETAIL_ANSWER: In the 1975 North Vietnamese legislative election, the Vietnamese Fatherland Front won all 424 seats.
SHORT_ANSWER: The Vietnamese Fatherland Front won all 424 seats.
"""

markdown_Freetqa_example2 = """
Table:
| Title           | Year    |  Role   | Notes |
|-----------------|---------|---------|-------|
|  Doctor Who     |   2008  | Ross Jenkins | "The Sontartagem", "The Posion Sky" |
| Demons          |  2009   | Luke Rutherford-Van Helsing | Main role            |
|  Trinity        |  2009   | Lord Dorian Gaudain         | Main role            |

QUESTION: What roles did Christian Cooke play in 2009?
DETAIL_ANSWER: During 2009, Christian Cooke played the lead role of Luke Rutherford in Demons & Lord Dorian Gaudain in Trinity.
SHORT_ANSWER: During 2009, Christian Cooke played the lead role of Luke Rutherford in Demons & Lord Dorian Gaudain in Trinity.
"""

markdown_Freetqa_examples = [markdown_Freetqa_example1, markdown_Freetqa_example2]


#==================================================markdown_Caption_examples
markdown_Caption_example1 = """
Table:
|party |candidate |total votes |% of total votes |
|---|---|---|---|
|Montreal Island Citizens Union |(x) Gérald Tremblay |202,302 |53.73 |
|Vision Montreal |Pierre Bourque |136,769 |36.32 |
|Projet Montréal |Richard Bergeron |32,126 |8.53 |
|White Elephant Party |Michel Bédard |5,329 |1.42 |
|Total valid votes |Total valid votes |376,526 |100 |

QUESTION: Given this table, please provide a detailed summary.
DETAIL_ANSWER: Gérald Tremblay led with 53.73% of the votes (202,302), followed by Pierre Bourque with 36.32% (136,769). Richard Bergeron received 8.53% (32,126), and Michel Bédard got 1.42% (5,329). The total valid votes were 376,526.
SHORT_ANSWER: Gérald Tremblay led with 53.73% of the votes (202,302), followed by Pierre Bourque with 36.32% (136,769). Richard Bergeron received 8.53% (32,126), and Michel Bédard got 1.42% (5,329). The total valid votes were 376,526.
"""

markdown_Caption_examples = [markdown_Caption_example1]
