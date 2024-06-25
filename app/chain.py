from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo")

db_user = "root"
db_password = "evolutivo"
db_host = ""
db_name = "classicmodels"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

execute_query = QuerySQLDataBaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()
generate_query = create_sql_query_chain(llm, db)
chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)