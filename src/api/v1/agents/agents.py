import os
import json
import cohere

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.tools.tools import RAGState, vector_search_node
from src.core.db import get_sql_database

load_dotenv()


from langchain_core.runnables import RunnableConfig
from datetime import datetime, timezone


def _langsmith_config(
    run_name: str,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """
    Builds LangSmith tracing config for LangChain runnable calls.

    Important:
    - metadata is visible in LangSmith traces.
    - metadata is NOT sent to the LLM as prompt content.
    """

    base_metadata = {
        "app": "northstar-credit-card-spend-summarizer",
        "module": "agent.py",
        "environment": os.getenv("APP_ENV", "local"),
        "openai_model": os.getenv("OPENAI_CHAT_MODEL"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if metadata:
        base_metadata.update(metadata)

    return {
        "run_name": run_name,
        "tags": tags or ["northstar", "rag", "credit-card"],
        "metadata": base_metadata,
    }


def normalize_chat_history(chat_history: list = None) -> list:
    """
    Normalizes chat history from Pydantic objects or dictionaries.

    Expected output:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """

    chat_history = chat_history or []
    normalized = []

    for msg in chat_history:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)

        if role is None and isinstance(msg, dict):
            role = msg.get("role")

        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if role in ["user", "assistant"] and content:
            normalized.append({"role": role, "content": str(content)})

    return normalized


# ── Helper: build the OpenAI LLM ──────────────────────────────────────────────


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ── Helper: initial state ─────────────────────────────────────────────────────
def _initial_state(query: str, chat_history: list = None) -> RAGState:
    return {
        "query": query,
        "chat_history": chat_history or [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "",
        "generated_sql": "",
        "sql_result": "",
        "score": 0.0,
        "reasoning": "",
        "sufficient": False,
        "answer_score": 0.0,
        "retry_count": 0,
    }


# ── Pydantic Models ───────────────────────────────────────────────────────────


class AnswerEvaluation(BaseModel):
    score: float
    reasoning: str
    sufficient: bool


class QueryToolInput(BaseModel):
    question: str = Field(
        ...,
        description="The user question to answer using this tool.",
    )


# ── Node NL2SQL: Translate query to SQL → Execute → Summarise ─────────────────


def nl2sql_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    db = get_sql_database()

    # ── Step 1: Generate SQL using the LLM + live schema ────────────────────
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a PostgreSQL expert for the NorthStar Bank Credit Card Spend Summarizer project.

Given the database schema below, write a single valid PostgreSQL SELECT query that answers the user's question.

Rules:
- Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
- Use only the tables and columns present in the provided schema.
- Generate only SELECT queries.
- Do NOT generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any DML/DDL statements.
- Never invent tables, columns, joins, values, rates, thresholds, or calculations.
- Always add a LIMIT clause with max 50 rows for non-aggregate/detail queries.
- Do not add LIMIT for aggregate queries unless the question asks for top N results.
- Use PostgreSQL syntax only.
- Never use SELECT * unless the user explicitly asks to view raw rows.
- Use clear aliases for computed fields.
- Avoid unnecessary joins.
- Use NULLIF where division by zero may occur.

Core Business Rules:
- Spend analytics must use txn_type = 'purchase' unless the user explicitly asks otherwise.
- Billing-month calculations must use billing_statements.start_date and billing_statements.end_date.
- Do not assume calendar month boundaries when billing statement dates are available.
- International spend must use is_international = TRUE.
- Business constants such as reward value, reward rates, fee-waiver thresholds, forex markup, charges, and product limits must be read from database/config/rules tables when such tables or columns exist in the schema.
- Do not hardcode reward values, fee-waiver thresholds, rates, charges, or limits.
- If the required business constant is not available in the schema, return a query using only available data instead of inventing the missing value.

Project Database Context:
The project includes tables such as:
- customers
- credit_cards
- card_transactions
- reward_transactions
- billing_statements

Use only the tables and columns actually present in the provided schema.

Common Query Intent Guidance:

1. Monthly spend summary:
- Join billing_statements with card_transactions using card_id and billing period.
- Use billing_statements.start_date and billing_statements.end_date for the requested billing_month.
- Sum purchase amounts and count purchase transactions.

2. Category breakdown:
- Use txn_type = 'purchase'.
- GROUP BY category_name.
- Return category_name, transaction count, total spend, percentage of total spend, and reward points earned if the columns exist.
- Use NULLIF for percentage calculations.

3. Top merchants:
- Use txn_type = 'purchase'.
- GROUP BY merchant_name.
- ORDER BY total spend DESC.
- LIMIT 5 unless the user asks for a different number.

4. International spend:
- Filter is_international = TRUE.
- Use txn_type = 'purchase' unless the user asks to include refunds, fees, or all international transactions.
- Include original_currency and original_amount for transaction-level results if those columns exist.

5. Reward points:
- For monthly points earned, use billing_statements.reward_pts_earned or transaction-level reward point columns if available.
- For available reward balance, use the relevant reward balance column from the card/account table if available.
- For INR value, use a conversion rate only if it exists in the schema or an available rules/config table; do not hardcode it.

6. Month-over-month comparison:
- Compare purchase spend for the requested billing_month against the previous billing month.
- Prefer billing_statements totals if available.
- Otherwise aggregate card_transactions using billing statement periods.

7. Fee waiver:
- Use year-to-date purchase spend from card_transactions.
- Use fee-waiver threshold only if it exists in the schema or an available rules/config table.
- Do not hardcode fee-waiver thresholds.
- If threshold is available, return ytd_spend, fee_waiver_target, and remaining_to_waiver using GREATEST(0, threshold - ytd_spend).

8. Statement summary:
- Use billing_statements for opening balance, total purchases, payments, fees, refunds, closing balance, minimum due, due date, and reward points earned when those columns exist.

Text Search Rules:
- For text searches on merchant names, customer names, category names, or card variants, split multi-word phrases into meaningful keywords.
- Use ILIKE with individual keywords.
- Search only across relevant text columns present in the schema, such as customer name, card variant, merchant name, or category name.
- Do not search a full multi-word phrase as a single ILIKE pattern unless the user explicitly asks for an exact phrase match.

Date Handling Rules:
- If the user provides a billing month like "March 2026", convert it to '2026-03'.
- If the user asks for "this month" or "last month" and no reference date is supplied, use the latest billing_month available in billing_statements for the relevant card.
- If card_id is provided, resolve billing months for that card only.
- If no card_id is provided but the question requires one, generate the safest possible query using available filters and limits.

Database schema:
{schema}""",
            ),
            ("human", "Question: {question}"),
        ]
    )

    sql_chain = sql_prompt | llm
    raw_sql = sql_chain.invoke(
        {
            "schema": schema_info,
            "question": state["query"],
        },
        config=_langsmith_config(
            run_name="nl2sql_generate_sql",
            metadata={
                "node": "nl2sql_node",
                "step": "generate_sql",
                "route": "sql_lookup",
                "query_type": "customer_database_query",
                "has_chat_history": bool(state.get("chat_history")),
                "chat_history_count": len(state.get("chat_history", [])),
            },
            tags=[
                "northstar",
                "nl2sql",
                "sql-generation",
                "sql_lookup",
            ],
        ),
    )

    content = raw_sql.content

    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )

    generated_sql = content.strip().strip("```").strip()

    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()

    print(f"[nl2sql_node] Generated SQL:\n{generated_sql}")

    # ── Step 2: Execute SQL ──────────────────────────────────────────────────
    try:
        sql_result: str = db.run(generated_sql)
    except Exception as exc:
        sql_result = f"SQL execution error: {exc}"

    print(f"[nl2sql_node] Raw result truncated: {str(sql_result)[:200]}")

    # ── Step 3: Summarise into AIResponse ────────────────────────────────────
    structured_llm = llm.with_structured_output(AIResponse)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful data analyst.

Answer the user's question using the SQL query results below.

Rules:
- Be concise.
- Format numbers and lists clearly.
- If the SQL result contains an error, explain that the database query could not be completed.
- Do not invent missing values.
- Set policy_citations to "N/A".
- Set page_no to "N/A".
- Set document_name to "credit_card_summary_seed_data_db".""",
            ),
            (
                "human",
                """Question:
{query}

SQL Used:
{sql}

Query Results:
{result}""",
            ),
        ]
    )

    chain = answer_prompt | structured_llm

    answer = chain.invoke(
        {
            "query": state["query"],
            "sql": generated_sql,
            "result": sql_result,
        },
        config=_langsmith_config(
            run_name="nl2sql_summarize_result",
            metadata={
                "node": "nl2sql_node",
                "step": "summarize_sql_result",
                "query": state["query"],
                "generated_sql": generated_sql,
                "route": "sql_lookup",
            },
            tags=["northstar", "nl2sql", "sql-answer"],
        ),
    )

    print("[nl2sql_node] Answer generated.")

    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["page_no"] = "N/A"
    response["document_name"] = "credit_card_summary_seed_data_db"
    response["sql_query_executed"] = generated_sql

    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response,
    }


# ── Node 2: Rerank ────────────────────────────────────────────────────────────


def rerank_node(state: RAGState) -> RAGState:
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    docs = state["retrieved_docs"]

    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=10,
    )

    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")

    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i + 1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}"
        )

    return {
        **state,
        "reranked_docs": reranked_docs,
    }


# ── Node 3: Generate Answer ───────────────────────────────────────────────────


def generate_answer_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    context = "\n\n".join(
        [
            (
                f"[Source: {doc.metadata.get('source_file', 'unknown')} | "
                f"Page: {doc.metadata.get('page_number', -1) + 1 if doc.metadata.get('page_number') is not None else '?'}]\n"
                f"{doc.page_content}"
            )
            for doc in state["reranked_docs"]
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant for the NorthStar Bank Credit Card Spend Summarizer project.

Answer the user's question using only the provided context from the credit card product guide / knowledge base.

The context may include information about:
- Card variants and features
- Annual fees, fee waiver rules, and credit limit ranges
- Spend categories and transaction classification
- Billing cycles, statements, payment due dates, and grace periods
- Reward points earn rates, redemption value, and expiry rules
- Fees, charges, forex markup, finance costs, and late payment slabs
- Spend summary business rules and output format
- Credit limit management rules
- Sample customer scenarios for testing

Answering rules:
- Use only the provided context. Do not use outside knowledge.
- Do not invent card features, fees, limits, reward rates, thresholds, dates, or calculations.
- If the answer is not present in the context, clearly say that the provided context does not contain enough information.
- If the question asks about actual customer spend, transactions, reward balance, MoM comparison, or fee waiver status for a specific card, explain that this requires database lookup rather than document-only context.
- If the question asks about rules or definitions, answer from the document context.
- If multiple chunks mention the same rule, consolidate them into one clear answer.
- If chunks contain conflicting values, prefer the chunk with the latest effective date or version if available, and explicitly mention the difference.
- Keep answers concise, business-friendly, and suitable for an internal banking assistant.

Citation rules:
- document_name: comma-separated list of EVERY source document you used.
- page_no: comma-separated page numbers, aligned with the documents above.
- policy_citations: readable citation combining each document and its page, for example: "KB_Credit_Card_Spend_Summarizer.pdf, Page 5".
- Always cite every document/page used to answer the question.
- If no relevant context is found, leave citation fields empty or state "No relevant source found" based on your response schema.""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{query}""",
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "context": context,
            "query": state["query"],
        },
        config=_langsmith_config(
            run_name="document_generate_answer",
            metadata={
                "node": "generate_answer_node",
                "step": "generate_document_answer",
                "route": "document_lookup",
                "retrieved_doc_count": len(state.get("retrieved_docs", [])),
                "reranked_doc_count": len(state.get("reranked_docs", [])),
                "has_context": bool(context.strip()),
                "context_char_count": len(context),
                "has_chat_history": bool(state.get("chat_history")),
                "chat_history_count": len(state.get("chat_history", [])),
            },
            tags=[
                "northstar",
                "rag",
                "document-answer",
                "document_lookup",
            ],
        ),
    )

    print("[generate_answer_node] Answer generated.")

    return {
        **state,
        "response": result.model_dump(),
    }


# ── Node 4: Evaluate Answer ───────────────────────────────────────────────────


def evaluate_answer_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    evaluator = llm.with_structured_output(AnswerEvaluation)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an answer quality evaluator.

Evaluate how well the generated answer addresses the user's question and score from 0 to 100.

Consider:
- Completeness
- Accuracy
- Relevance
- Whether the user's question was fully answered

Mark sufficient=true only if score >= 90.""",
            ),
            (
                "human",
                """User Question:
{question}

Generated Answer:
{answer}""",
            ),
        ]
    )

    chain = prompt | evaluator

    result = chain.invoke(
        {
            "question": state["query"],
            "answer": state["response"].get("answer", ""),
        },
        config=_langsmith_config(
            run_name="evaluate_answer_quality",
            metadata={
                "node": "evaluate_answer_node",
                "step": "answer_evaluation",
                "query": state["query"],
                "retry_count": state.get("retry_count", 0),
            },
            tags=["northstar", "evaluation", "answer-quality"],
        ),
    )

    print(
        f"[evaluate_answer_node] Score={result.score} "
        f"Sufficient={result.sufficient}"
    )

    return {
        **state,
        "answer_score": result.score,
        "reasoning": result.reasoning,
        "sufficient": result.sufficient,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def evaluation_router(state: RAGState) -> str:
    score = state.get("answer_score", 0)
    retries = state.get("retry_count", 0)

    if score >= 90:
        return "accepted"

    if retries >= 3:
        return "accepted"

    return "retry"


# ── Tool 1: SQL Lookup ────────────────────────────────────────────────────────


@tool("sql_lookup", args_schema=QueryToolInput)
def sql_lookup(question: str, chat_history: list = None) -> str:
    """
    Use this tool for customer-specific, card-specific, account-specific,
    transaction-specific, billing-specific, reward-specific, or computed SQL answers.
    """

    print("[sql_lookup] Tool called.")

    effective_chat_history = normalize_chat_history(chat_history)

    tool_state = _initial_state(query=question, chat_history=effective_chat_history)

    result_state = nl2sql_node(tool_state)

    return json.dumps(
        {
            "tool": "sql_lookup",
            "answer": result_state.get("response", {}),
            "generated_sql": result_state.get("generated_sql", ""),
            "sql_result": result_state.get("sql_result", ""),
        },
        default=str,
    )


# ── Tool 2: Document Lookup ───────────────────────────────────────────────────
@tool("document_lookup", args_schema=QueryToolInput)
def document_lookup(question: str, chat_history: list = None) -> str:
    """
    Use this tool for product guide, policy, rules, fees, rewards,
    billing logic, spend summary rules, card features, edge cases,
    expected output formats, and knowledge-base/document questions.
    """

    print("[document_lookup] Tool called.")

    effective_chat_history = normalize_chat_history(chat_history)

    tool_state = _initial_state(query=question, chat_history=effective_chat_history)

    max_retries = 3

    for _ in range(max_retries):
        tool_state["retrieved_docs"] = []
        tool_state["reranked_docs"] = []

        tool_state = vector_search_node(tool_state)
        tool_state = rerank_node(tool_state)
        tool_state = generate_answer_node(tool_state)
        tool_state = evaluate_answer_node(tool_state)

        if evaluation_router(tool_state) == "accepted":
            break

    return json.dumps(
        {
            "tool": "document_lookup",
            "answer": tool_state.get("response", {}),
            "answer_score": tool_state.get("answer_score", 0),
            "retry_count": tool_state.get("retry_count", 0),
            "reasoning": tool_state.get("reasoning", ""),
            "sufficient": tool_state.get("sufficient", False),
        },
        default=str,
    )


# ── Dynamic Tool Agent Node ───────────────────────────────────────────────────
# This replaces the old static router_node.
# The LLM can call sql_lookup, document_lookup, or both.


def build_conversation_context(chat_history: list = None) -> str:
    """
    Converts chat_history into readable conversation text for the LLM.

    Input format:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]

    Output format:
    User: ...
    Assistant: ...
    """

    chat_history = chat_history or []
    conversation_context_parts = []

    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        if not content:
            continue

        if role == "user":
            conversation_context_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_context_parts.append(f"Assistant: {content}")

    if not conversation_context_parts:
        return "No previous conversation."

    return "\n".join(conversation_context_parts)


def rewrite_followup_question(state: RAGState) -> str:
    """
    Rewrites follow-up questions into standalone questions using chat history.

    Example:
    Previous answer: Robert Clarke, Customer ID C-1003 has highest credit limit.
    Current query: what is his available limit as of now?
    Rewritten query: What is the available credit limit for Robert Clarke, Customer ID C-1003, as of now?
    """

    llm = _get_llm()

    query = state["query"]
    chat_history = state.get("chat_history", []) or []
    conversation_context = build_conversation_context(chat_history)

    if not chat_history:
        return query

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You rewrite follow-up user questions into standalone questions.

Rules:
- Use previous conversation to resolve pronouns and references.
- Resolve words like his, her, it, that, this, same, above, previous, that customer, that card, that category.
- Preserve important identifiers such as customer_id, card_id, account_id, customer name, merchant, month, category, and date.
- Do not answer the question.
- Only return the rewritten standalone question.
- If the question is already standalone, return it unchanged.

Examples:
Previous conversation:
User: Which customer has the highest credit limit?
Assistant: The customer with the highest credit limit is Robert Clarke (Customer ID: C-1003).

Current question:
What is his available limit as of now?

Standalone question:
What is the available credit limit for Robert Clarke, Customer ID C-1003, as of now?""",
            ),
            (
                "human",
                """Previous conversation:
{conversation_context}

Current question:
{query}

Standalone question:""",
            ),
        ]
    )

    chain = rewrite_prompt | llm

    rewritten = chain.invoke(
        {
            "conversation_context": conversation_context,
            "query": query,
        },
        config=_langsmith_config(
            run_name="rewrite_followup_question",
            metadata={
                "node": "rewrite_followup_question",
                "step": "query_rewrite",
                "original_query": query,
                "has_chat_history": bool(chat_history),
            },
            tags=["northstar", "query-rewrite", "conversation-memory"],
        ),
    )

    rewritten_text = getattr(rewritten, "content", str(rewritten)).strip()

    print(f"[rewrite_followup_question] Original query: {query}")
    print(f"[rewrite_followup_question] Rewritten query: {rewritten_text}")

    return rewritten_text or query


def dynamic_tool_agent_node(state: RAGState) -> RAGState:
    llm = _get_llm()

    chat_history = state.get("chat_history", []) or []
    conversation_context = build_conversation_context(chat_history)
    contextual_query = rewrite_followup_question(state)

    print(f"[dynamic_tool_agent_node] Original query: {state['query']}")
    print(f"[dynamic_tool_agent_node] Contextual query: {contextual_query}")

    tools = [sql_lookup, document_lookup]
    tool_map = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    # ---------------------------------------------------------------------
    # Build conversation context from chat_history
    # ---------------------------------------------------------------------
    chat_history = state.get("chat_history", []) or []

    conversation_context_parts = []

    for msg in chat_history:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )

        if not content:
            continue

        if role == "user":
            conversation_context_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_context_parts.append(f"Assistant: {content}")

    conversation_context = "\n".join(conversation_context_parts)

    if not conversation_context:
        conversation_context = "No previous conversation."

    # ---------------------------------------------------------------------
    # Planner prompt with chat history
    # ---------------------------------------------------------------------
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a dynamic tool-calling supervisor for the NorthStar Bank Credit Card Spend Summarizer project.

You have access to two tools:

1. sql_lookup

Use sql_lookup when the user asks for actual data from the PostgreSQL database, including:
- customer details
- card details
- card_id or customer_id specific questions
- transactions
- billing statements
- outstanding amount
- available credit limit
- current available limit
- remaining limit
- usable limit
- credit limit minus outstanding balance
- minimum due
- due date
- reward balance
- spend totals
- category-wise spend
- top merchants
- monthly spend
- international spend
- fee waiver eligibility for a specific card/customer
- computed values for a specific account/card/customer

2. document_lookup

Use document_lookup when the user asks for knowledge-base or product-guide information, including:
- card variants and features
- reward rules
- annual fee rules
- fee waiver policy
- forex markup rules
- billing cycle rules
- grace period
- finance charges
- late payment slabs
- spend summary business rules
- category definitions
- edge cases
- expected output format
- implementation guidance

Important routing behavior:
- You may call only sql_lookup.
- You may call only document_lookup.
- You may call both tools if the user question needs both actual database facts and document/policy explanation.
- Prefer calling both tools when a complete answer requires both customer/account data and product-guide rules.
- Do not answer directly if a tool can answer the question.

Conversation memory rules:
- Use the previous conversation to understand follow-up questions.
- Resolve references like "it", "that", "this", "same", "above", "those", "that card", "that customer", "that category", "that merchant", and "previous month".
- If the current question is a follow-up, rewrite it mentally using the previous conversation before selecting tools.
- If the user asks for actual available limit, current limit, remaining limit, outstanding amount, or customer/card-specific values, always use sql_lookup.
- Do not use document_lookup alone for these questions.
- Pass the relevant standalone question to the selected tool.
- Do not deviate from the active conversation topic.

Examples:
- "What is the spend for card CC-881001 in March 2026?" -> sql_lookup only.
- "What is the forex markup rule?" -> document_lookup only.
- "what is his available limit" -> sql_lookup because his refers to previous customer in the conversation.
- "Is card CC-881001 eligible for annual fee waiver and explain the rule?" -> sql_lookup and document_lookup.
- "How many reward points did Sarah earn and what is the redemption value?" -> sql_lookup and document_lookup if redemption value is in the document.
- Previous: "What is my highest spend category?" Current: "Which merchants contributed to it?" -> sql_lookup because "it" refers to the previous category.""",
            ),
            (
                "human",
                """Previous conversation:
        {conversation_context}

        Original user question:
        {original_query}

        Standalone resolved question:
        {query}

        Use the standalone resolved question for tool selection and tool arguments.""",
            ),
        ]
    )

    planner_chain = planner_prompt | llm_with_tools

    planner_response = planner_chain.invoke(
        {
            "query": contextual_query,
            "original_query": state["query"],
            "conversation_context": conversation_context,
        },
        config=_langsmith_config(
            run_name="dynamic_tool_planner",
            metadata={
                "node": "dynamic_tool_agent_node",
                "step": "tool_planning",
                "original_query": state["query"],
                "contextual_query": contextual_query,
                "has_chat_history": bool(chat_history),
            },
            tags=["northstar", "tool-router", "planner"],
        ),
    )

    tool_calls = getattr(planner_response, "tool_calls", []) or []

    tool_results = []

    if not tool_calls:
        print(
            "[dynamic_tool_agent_node] No tool call produced. Falling back to document_lookup."
        )

        fallback_result = document_lookup.invoke(
            {
                "question": contextual_query,
                "chat_history": chat_history,
            }
        )

        tool_results.append(
            {
                "tool": "document_lookup",
                "result": fallback_result,
            }
        )

    else:
        print(f"[dynamic_tool_agent_node] Tool calls detected: {len(tool_calls)}")

        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})

            if tool_name not in tool_map:
                print(f"[dynamic_tool_agent_node] Unknown tool skipped: {tool_name}")
                continue

            if isinstance(tool_args, dict):
                question = tool_args.get("question", contextual_query)
            else:
                question = state["query"]

            print(f"[dynamic_tool_agent_node] Calling tool: {tool_name}")

            result = tool_map[tool_name].invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )

            tool_results.append(
                {
                    "tool": tool_name,
                    "result": result,
                }
            )

    structured_llm = llm.with_structured_output(AIResponse)

    # ---------------------------------------------------------------------
    # Final synthesis prompt with chat history
    # ---------------------------------------------------------------------
    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the final answer synthesizer for the NorthStar Bank Credit Card Spend Summarizer project.

You are given:
- The previous conversation
- The current user question
- Outputs from one or more tools

Tool output meanings:
- sql_lookup may contain actual database results and generated SQL.
- document_lookup may contain policy/product-guide answers and citations.

Rules:
- Answer the user's current question clearly and concisely.
- Use the previous conversation to resolve follow-up references.
- If the user says "it", "that", "this", "same", "above", "those", or similar, infer the reference from the previous conversation.
- If SQL data is available, use it as the source of truth for actual customer/card/account/transaction values.
- If document context is available, use it as the source of truth for policy, rule, product-guide, fee, reward, and billing explanations.
- If both tools were used, combine them into one complete answer.
- Do not invent values that are not present in the tool outputs.
- Preserve document citations when provided by document_lookup.
- For SQL-only answers, set policy_citations to "N/A", page_no to "N/A", and document_name to "credit_card_summary_seed_data_db".
- For mixed SQL + document answers, include document citations from document_lookup and mention SQL was used for customer/account facts.
- If a tool returned an error, explain the limitation clearly without fabricating an answer.
- Do not deviate from the active conversation topic.""",
            ),
            (
                "human",
                """Previous conversation:
{conversation_context}

Current user question:
{query}

Tool outputs:
{tool_outputs}""",
            ),
        ]
    )

    final_chain = synthesis_prompt | structured_llm

    final_answer = final_chain.invoke(
        {
            "query": state["query"],
            "conversation_context": conversation_context,
            "tool_outputs": json.dumps(tool_results, default=str),
        },
        config=_langsmith_config(
            run_name="final_answer_synthesis",
            metadata={
                "node": "dynamic_tool_agent_node",
                "step": "final_synthesis",
                "query": state["query"],
                "tools_used": [item.get("tool") for item in tool_results],
                "tool_count": len(tool_results),
            },
            tags=["northstar", "final-answer", "synthesis"],
        ),
    )

    response = final_answer.model_dump()

    sql_queries = []
    sql_results = []

    for item in tool_results:
        if item.get("tool") == "sql_lookup":
            try:
                parsed = json.loads(item.get("result", "{}"))

                generated_sql = parsed.get("generated_sql", "")
                sql_result = parsed.get("sql_result", "")

                if generated_sql:
                    sql_queries.append(generated_sql)

                if sql_result:
                    sql_results.append(sql_result)

            except Exception as exc:
                print(
                    f"[dynamic_tool_agent_node] Could not parse SQL tool result: {exc}"
                )

    if sql_queries:
        response["sql_query_executed"] = "\n\n".join(sql_queries)

    print("[dynamic_tool_agent_node] Final synthesized answer generated.")

    return {
        **state,
        "response": response,
        "generated_sql": "\n\n".join(sql_queries),
        "sql_result": "\n\n".join(sql_results),
    }


# ── Build the LangGraph ───────────────────────────────────────────────────────
# New graph:
#
#   dynamic_tool_agent ──► END
#
# The dynamic_tool_agent internally decides whether to call:
# - sql_lookup
# - document_lookup
# - both


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("dynamic_tool_agent", dynamic_tool_agent_node)

    graph.set_entry_point("dynamic_tool_agent")
    graph.add_edge("dynamic_tool_agent", END)

    compiled_agent = graph.compile()

    os.makedirs("references", exist_ok=True)

    try:
        mermaid_code = compiled_agent.get_graph().draw_mermaid()
        with open("rag_graph.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_code)
    except Exception as e:
        print(f"Graph generation skipped: {e}")

    return compiled_agent


# Compile once at module load — reused across all requests
rag_graph = build_rag_graph()


# ── Async Public Entrypoint ───────────────────────────────────────────────────
# This returns the final structured response as SSE.
# It avoids streaming intermediate planner/tool model tokens.


async def run_search_agent(query: str, chat_history: list = None):
    """
    Runs the RAG graph with current query + previous chat history.

    chat_history expected format:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """

    chat_history = chat_history or []

    # Normalize chat_history because it may contain Pydantic objects or dicts
    normalized_chat_history = []

    for msg in chat_history:
        role = getattr(msg, "role", None) or msg.get("role")
        content = getattr(msg, "content", None) or msg.get("content")

        if role in ["user", "assistant"] and content:
            normalized_chat_history.append({"role": role, "content": content})

    initial_state: RAGState = _initial_state(
        query=query, chat_history=normalized_chat_history
    )

    try:
        final_state = await rag_graph.ainvoke(initial_state)

        response_payload = final_state.get("response", {})

        yield f"data: {json.dumps(response_payload, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as exc:
        error_payload = {
            "answer": f"An error occurred while processing the query: {str(exc)}",
            "policy_citations": "",
            "page_no": "",
            "document_name": "",
        }

        yield f"data: {json.dumps(error_payload, default=str)}\n\n"
        yield "data: [DONE]\n\n"
