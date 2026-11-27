"""Customer service with fixture data and a local in-memory checkpointer."""

import json
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search_faq(query: str) -> str:
    """Search the FAQ knowledge base. Suitable for answering questions about product usage, policies, and processes."""
    faq_data = {
        "refund": "Refund policy: no-questions-asked refunds within 7 days; original packaging required. Contact customer service to apply.",
        "shipping": "Standard shipping: 1–3 business days; delayed during holidays. Express shipping available.",
        "warranty": "Products purchased through official channels come with a 1-year official warranty. Screen damage is not covered.",
        "discount": "New users get 20% off their first order. Members can use points to offset payment.",
        "payment": "Accepts credit cards, PayPal, and bank transfers. Cash on delivery not supported.",
    }

    for keyword, answer in faq_data.items():
        if keyword in query.lower():
            return answer

    return "No relevant FAQ found. We recommend contacting live customer support for more help."

@tool
def check_order(order_id: str) -> str:
    """
    Query order status and shipping information.
    Input an order number (e.g., ORD-12345678) to get order details.
    """
    orders = {
        "ORD-12345678": {
            "status": "Shipped",
            "items": "Python Programming Book × 1",
            "amount": 29.99,
            "shipping": "FedEx: FX1234567890, expected delivery tomorrow"
        },
        "ORD-87654321": {
            "status": "Pending Shipment",
            "items": "AI Development Course × 1",
            "amount": 99.00,
            "shipping": "Expected to ship tomorrow"
        }
    }

    order = orders.get(order_id)
    if order:
        return json.dumps(order, ensure_ascii=False)
    return f"Order {order_id} not found. Please verify the order number."

@tool
def draft_complaint(
    order_id: str,
    complaint_type: str,
    description: str
) -> str:
    """
    Draft an after-sales complaint for review.
    complaint_type: refund request / quality issue / shipping issue / other
    """
    return json.dumps(
        {"order_id": order_id, "type": complaint_type, "description": description},
        ensure_ascii=False,
    )


@tool
def recommend_products(user_need: str) -> str:
    """Recommend suitable products based on user needs."""
    catalog = [
        {"name": "Python: From Beginner to Expert", "price": 29.99, "tag": "programming python beginner"},
        {"name": "AI Agent Hands-On Course", "price": 99.00, "tag": "ai machine-learning agent"},
        {"name": "LangChain Practical Tutorial", "price": 69.00, "tag": "langchain python ai"},
        {"name": "Data Analysis Complete Guide", "price": 49.99, "tag": "data-analysis pandas excel"},
    ]

    # Simple keyword matching
    results = []
    for item in catalog:
        if any(keyword in user_need.lower() for keyword in item["tag"].split()):
            results.append(f"• {item['name']} - ${item['price']}")

    if results:
        return "Based on your needs, here are our recommendations:\n" + "\n".join(results)
    return "No exact matches found. Please describe your needs in more detail."



def main() -> None:
    agent = create_agent(
        ChatOpenAI(model=os.getenv("CUSTOMER_SERVICE_MODEL", "gpt-4.1-mini")),
        tools=[search_faq, check_order, draft_complaint, recommend_products],
        system_prompt=(
            "You are a customer service assistant for a fictional store. "
            "Use the tools for factual answers, and describe draft complaints as drafts."
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "customer-demo"}, "recursion_limit": 20}
    print("Customer service demo. Type 'quit' to exit.")
    while True:
        try:
            message = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() == "quit":
            break
        if message:
            state = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
            print("Assistant:", state["messages"][-1].content)


if __name__ == "__main__":
    main()
