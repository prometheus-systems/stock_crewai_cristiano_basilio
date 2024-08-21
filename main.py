#iimportação das bibliotecas
import json
import os
from datetime import datetime
import requests
import yfinance as yf
from crewai import Agent,Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st

#CRIANDO FINANCE YAHOO TOOLS
def fetch_stock_price(ticket):
  stocks = yf.download(ticket, start='2023-08-21', end='2024-08-21')
  return stocks

yahoo_finance_tools = Tool(
    name = "Yahoo Finance Tools",
    description = "Fetch stock prices for {ticket} from the last year about especific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)

#IMPORTANDO OPENAI LLM GPT
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
#"sk-proj-bn9bCANK2FZ7calz5wHST3BlbkFJlqR39Gl1XUugv7BVVzGu"
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=12000)

stock_price_agent = Agent(
    role="Senior stock price analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""you're highly experienced in analyzing the price of an especific stock
    and make predictions about its future price.""",
    verbose=False,
    llm=llm,
    max_iterations=5,
    memory=True,
    allow_delegation=True,
    tools=[yahoo_finance_tools]
)

getStockPrice = Task(
    description="Analyse the stock {ticket} price history and create a trend analyses up, down or sideways",
    expected_output="""Specify the current trend stock price - up, down or sideways.
    eg. stock = APL, price UP.""",
    agent=stock_price_agent
)

#IMPORTANDO A TOOLS DE SEARCH
search_tools = DuckDuckGoSearchResults(backend="news", num_results=10)

newAnalyst = Agent(
    role="Stock news analyst",
    goal="""Create a short summary of market news related to de stock {ticket} company. Specify de current trend - up, down or sideways with the news context.
    For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""you're highly experienced in analyzing the market trends and news and have tracked assets for more 10 years.

    You're also master level analysts in the tradicional markets and have deep understand of human psychology.
    You understand news, theirs tittles and information, but you look at those with health dose of skepticism.
    You consider also the source of the news articles.""",
    verbose=False,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True
)

getNews = Task(
    description="""Take the stock and always include BTC to it (inf not request).
    Use de search tool to search each on individualy

    The current date is {datahora}

    Compose the results into a helpfull report""",
    expected_output="""A Summary of the overall market end on sentence summary for each request assets.
    Include a fear/greed score for each asset based the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTIONS>
    <FEAR/GREED SCORE>""",
    agent=newAnalyst
)

stockAnalistWrite = Agent(
    role="Senior stock analyst writer",
    goal="""Analyze trends price and news and write and unsighfull compelling in informative 3 paragraph long newsletter based on de stock report and price trend""",
    backstory="""You're widely accepted as the best stock analyst in the market.data=
    You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You anderstand macro factores and combine multipli theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinios when analysing anything """,
    verbose=False,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True
)

writeAnalyses = Task(
    description="""Use de stock price trend and stock news report to create analyses and write the newsletter about the {ticket} company that is brief and highligths the most important points. # Changed '{ticket}' to '{ticket}'
    Focus and stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary""",
    expected_output="""An eloquent 3 paragraphs newsletters formated as markdown in an easy readable manner. It should contain:
    - 3 bullets executive summary
    - introduction - set the overall picture and spike up the interest.
    - main part provides the meat of analyses including the news summary and fear/greed scores.
    - summary - key facts and concrete future trend prediction - up, down sideways """,
    agent=stockAnalistWrite,
    context=[getStockPrice,getNews]
)

crew = Crew(
    agents=[stock_price_agent,newAnalyst,stockAnalistWrite],
    tasks=[getStockPrice,getNews,writeAnalyses],
    verbose=False,
    process=Process.hierarchical,
    full_output=True,
    max_iterations=15,
    share_crew=False,
    manager_llm=llm
)
#crew.run()

import datetime

# Calculate the current datetime first
now = datetime.datetime.now()

# Format the datetime object as a string for interpolation
formatted_datetime = now.strftime('%Y-%m-%d %H:%M:%S')

#results = crew.kickoff(inputs={'ticket':'AAPL', 'datahora': formatted_datetime})
# Pass the formatted datetime string to the inputs dictionary


#results['final_output']

with st.sidebar:
    st.header('Stock Price Analysis')
    with st.form(key='resource_form'):
      topic = st.text_input("Select the ticket")
      submit_button = st.form_submit_button(label="Run research")

if submit_button:
  if not topic:
    st.error("Try again")
  else:
    results = crew.kickoff(inputs={'ticket':topic, 'datahora': formatted_datetime})
    st.subheader("Results:")
    st.write("Final Output:")
    st.write(results['final_output'])