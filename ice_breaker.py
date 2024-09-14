from typing import Tuple

from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
#from langchain_community.chat_models import ChatOllama # Just another open source model
from langchain_core.output_parsers import StrOutputParser

from output_parsers import summary_parser,Summary
from third_parties.linkedin import scrape_linkedin_profile

#importing agent
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break_with(name:str)-> Tuple[Summary,str]:
    linkedin_url=linkedin_lookup_agent(name=name) # agent will do this
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)


    prompt_template = """
    given the linkedin information {information} about a person that I want you to create:
    1. A short summary
    2. two interesting facts about them
    
    \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=prompt_template,
        partial_variables={"format_instructions":summary_parser.get_format_instructions()}
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro")  # no need to pass api key as parameter
    # llm = ChatOllama(model="llama3")

    chain = summary_prompt_template | llm | summary_parser

    result:Summary = chain.invoke(input={"information": linkedin_data})

    return result,linkedin_data.get("profile_pic_url")

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain")
    ice_break_with("Shivam Jain IIT Indore")
