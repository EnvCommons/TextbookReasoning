"""
Test agent for TextbookReasoning environment.

Tests the environment locally before deployment.
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    # Configuration
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "local/TextbookReasoning"  # Use "EnvCommons/textbookreasoning" for deployed
    SPLIT = "train"
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    # Initialize clients
    or_client = AsyncOpenReward()  # Remove base_url for deployed
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Get environment and tasks
    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in {SPLIT} split")
    # Test with first task
    task = tasks[0]
    print(task)

    async with environment.session(
        task=task,
        secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        # Get initial prompt
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False

        print(f"\nQuestion: {prompt[0].text[:200]}...")

        # Agent loop
        while not finished:
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list
            )

            input_list += response.output

            # Process tool calls
            for item in response.output:
                if item.type == "function_call":
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments))
                    )

                    reward = tool_result.reward
                    finished = tool_result.finished

                    # Add tool output to conversation
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result.blocks[0].text
                    })

                    print(f"\nTool: {item.name}")
                    submitted_answer = json.loads(str(item.arguments))['answer']
                    print(f"Submitted: {submitted_answer[:100]}...")
                    print(f"Reward: {reward:.3f}")
                    print(f"Grader response:\n{tool_result.blocks[0].text}")

                    if tool_result.finished:
                        finished = True
                        print("\nTask finished!")
                        break

            # Safety: break if no tool calls
            if not any(i.type == "function_call" for i in response.output):
                print("No tool call made, ending session")
                break


if __name__ == "__main__":
    asyncio.run(main())
