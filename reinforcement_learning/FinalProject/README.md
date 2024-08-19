# CSCN8020RL
Reinforcement Learning Project Report for Stock Market Forecast App: RL-DDQN, SARSA, and Q-Learning
Author: Tessa Nejla Ayvazoğlu
Student ID: 8686601
Date: 16/08/2024

I. DDQN: Detailed Analysis
II.Conclusion: Hold Decision for DDQN
III. Report on the Application of SARSA and Q-Learning on NVDA Stock Data
IV.Conclusion: Hold Decision Based on Q-Learning and SARSA Results
V. Stock Forecast App with AI LSTM
VI. Results of All Project

1. Architectural Flow Diagram
1.1. General Flow
Data Collection
Source: Yahoo Finance API
Output: Stock data (e.g., NVDA) saved as a CSV file
Environment Setup
Class: TradingEnvironment
Input: Stock data
Output: Trading actions and rewards
Agent Setup
Class: DDQNAgent
Networks: Online and Target Networks
Parameters: Learning rate, gamma, epsilon, etc.
Training Process
Training Loop: for episode in range(max_episodes + 1):
Action Selection: epsilon_greedy_policy
Experience Replay: memorize_transition and experience_replay
Reward Calculation: reward = Final NAV × (1 + Strategy Return)
Results Analysis
Performance Metrics: NAV, epsilon decay
Graphs: NAV comparison, epsilon decay

Conclusion: Hold Decision for DDQN
Performance Analysis and Rationale:
1. Agent Performance:
Net Asset Value (NAV) Stability: The DDQN agent's NAV remains stable over time. This indicates that the agent is demonstrating a consistent performance with its current strategies, unaffected by market fluctuations.
Market Volatility: The market values show high volatility, reflecting significant fluctuations. This suggests the need for cautious decision-making and a focus on risk management.
2. Risk Management:
Importance of Stable Performance: The stability of the agent's NAV suggests that its risk management strategies are effective and play a protective role against market volatility. This implies that maintaining current positions (Hold) might be a better strategy to avoid the potential risks associated with market changes.
3. Rationale for Hold Decision:
Uncertain Market Conditions: Given the high volatility and uncertainties in the market, holding current positions (Hold) may be a sensible strategy to avoid exposure to sudden price movements and reduce risk. The current market conditions suggest that maintaining existing positions could be less risky.
Alignment with Agent’s Strategy: The agent’s performance indicates that holding positions aligns well with the agent’s strategies and provides a lower-risk option compared to taking new actions.
In summary, considering the DDQN agent’s stable NAV performance and the high volatility of the market, holding current positions (Hold) is deemed the most prudent decision. This approach allows the investor to reduce risk and avoid being adversely affected by market fluctuations while continuing to follow the existing strategy.


II.Report on the Application of SARSA and Q-Learning on NVDA Stock Data
[ program name: RL_Stock_DDQN_V1.ipynb]
1. Introduction
This report presents the application and comparison of two popular Reinforcement Learning (RL) algorithms, SARSA and Q-Learning, on NVDA stock price data. The primary goal of this project was to develop an RL agent capable of making trading decisions based on historical price data, ultimately aiming to maximize the total reward, i.e., the financial gain accrued from trading activities.
2. Methodology
The RL agent was designed to navigate the NVDA stock price data and learn to make decisions on whether to buy, hold, or sell. The performance of the agent was evaluated based on the total accumulated reward over a series of trading sessions. Two different RL algorithms were employed:
SARSA (State-Action-Reward-State-Action)
Q-Learning
Both algorithms operate within the framework of Markov Decision Processes (MDP), with a slight difference in their update rules. The formulas governing these algorithms are provided below:


Conclusion: Hold Decision Based on Q-Learning and SARSA Results
1. Action and Reward Analysis:
Action Taken:
Action: 0 (Sit)
Reward: 0
The output indicates that the agent took the action to "sit" (which corresponds to holding the position), and the reward received was 0. This suggests that the agent's decision did not result in any tangible benefit or improvement under the given conditions.
2. Interpretation:
Negative or Zero Reward:
Reward of 0: This implies that the action did not lead to any positive outcome. In this scenario, the agent’s decision to "sit" did not yield any benefits. This could be because the current state did not present an immediate opportunity for buying or selling, or it might reflect a suboptimal decision based on the agent's current policy and state evaluation.
Action Evaluation:
Hold as Default Action: The choice to "sit" might be a default action when the agent does not identify any immediate advantageous moves. While this action did not result in rewards, it may be a conservative strategy in uncertain or volatile market conditions. A lack of positive reward suggests that the state was not favorable for other actions, or the current strategy needs refinement.
3. Conclusion:
Why "Hold" is Recommended:
Risk Mitigation: Given the absence of a positive reward and the potential suboptimal outcomes from other actions, maintaining the current positions (Hold) may be the most prudent decision. This approach minimizes the risk associated with potentially unfavorable market conditions or incorrect action choices.
Policy and Strategy Improvement: The zero reward highlights the need for continuous improvement in the agent's policy and learning parameters. Adjustments may be necessary to enhance decision-making and ensure that actions lead to positive outcomes in various market scenarios.
In summary, based on the results from Q-learning and SARSA, a "Hold" decision is recommended. The lack of positive rewards for other actions underscores the need for a cautious approach in uncertain market conditions. Maintaining existing positions helps manage risk while providing an opportunity to refine the agent's strategies for future decisions.


IV. STOCK FORECAST APP WITH AI LSTM: 

V. Results of All Project
Web Application for Stock Market Analysis
Our web application provides in-depth analysis of selected tickers, offering actionable insights to investors. For the specific case of NVIDIA (NVDA), the application has recommended a "HOLD" position. This recommendation aligns with the broader analysis indicating a high percentage of "HOLD" advice for this month across various analyses.
Key Findings:
NVIDIA (NVDA) Recommendation:
Recommendation: HOLD
Analysis: The deep analysis conducted for NVDA suggests maintaining the current position. This advice is based on the agent’s evaluation using Deep Q-Learning (DDQN), SARSA, and Q-learning, all of which converge on the recommendation to "HOLD."
General Analysis:
Market Trend: This month’s analyses consistently show a high percentage of "HOLD" recommendations across different methods.
Agent Performance: The DDQN, SARSA, and Q-learning models have been utilized to derive the recommendation for NVDA. Each method supports the "HOLD" position, indicating a stable recommendation across various reinforcement learning algorithms.
Conclusion:
The application effectively leverages reinforcement learning techniques to provide a consistent "HOLD" recommendation for NVDA, reflecting a cautious approach in the current market conditions. The use of DDQN, SARSA, and Q-learning models ensures robust and well-supported investment advice.

THANK YOU!

Contact info:https://www.linkedin.com/in/tessa-nejla-ayvazoglu/
                      nejlayvazoglu@gmail.com
                      https://github.com/TessaAyv79/CSCN8020RL

PorFolio: https://portfolio-eight-bice-81.vercel.app/personal.html
