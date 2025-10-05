import os, time, uuid
from typing import Optional, Literal, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AzureOpenAI
from openai import OpenAI
import logging

# ---------- Debug logging ----------
load_dotenv()
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nerv-backend")

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

logger.info(f"[BOOT] Azure endpoint: {AZURE_OPENAI_ENDPOINT or '(missing)'}")
logger.info(f"[BOOT] Azure deployment: {AZURE_OPENAI_DEPLOYMENT or '(missing)'}")
logger.info(f"[BOOT] Azure key present: {bool(AZURE_OPENAI_API_KEY)}")

if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
    logger.error("Missing Azure OpenAI envs (key/endpoint/deployment).")
    raise RuntimeError("Missing Azure OpenAI envs (key/endpoint/deployment).")

AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# MongoDB
MONGODB_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("DB_NAME", "nerv_interview")
logger.info(f"[BOOT] Mongo URI present: {bool(MONGODB_URI)}  DB: {DB_NAME}")
if not MONGODB_URI:
    logger.error("Missing MONGODB_URI.")
    raise RuntimeError("Missing MONGODB_URI.")
mongo = AsyncIOMotorClient(MONGODB_URI)
db = mongo[DB_NAME]
chats = db["chats"]

app = FastAPI(title="NERV Interview Backend", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*", "X-Conversation-Id"],
)

RoundType = Literal["technical", "project", "hr"]

class AskRequest(BaseModel):
    # Only two inputs from frontend
    emotion: str = Field(..., description="e.g., confident/nervous/…")
    last_answer: Optional[str] = None

class ProjectAskRequest(BaseModel):
    emotion: str = Field(..., description="e.g., confident/nervous/…")
    last_answer: Optional[str] = None
    projects: List[str] = Field(default=[], description="List of projects from resume")
    skills: List[str] = Field(default=[], description="List of skills from resume")

class HrAskRequest(BaseModel):
    emotion: str = Field(..., description="e.g., confident/nervous/…")
    last_answer: Optional[str] = None
    experiences: List[str] = Field(default=[], description="List of experiences from resume")
    achievements: List[str] = Field(default=[], description="List of achievements from resume")

class AskResponse(BaseModel):
    question: str
    round: RoundType
    conversation_id: str

class HistoryMessage(BaseModel):
    role: str
    content: str
    ts: int
    emotion: Optional[str] = None
    question_id: Optional[str] = None
    difficulty: Optional[str] = None

class HistoryResponse(BaseModel):
    conversation_id: str
    round: RoundType
    messages: List[HistoryMessage]

# -------- System prompts --------
COMMON_ANTI_HALLUCINATION = """
Follow these rules strictly:
- Do NOT invent facts. If info is missing or unclear, ask a short clarifying question.
- Base your question ONLY on the conversation history and the candidate's last answer.
- Never mention the word 'undefined'. If the last answer is unavailable, acknowledge briefly and ask a clarifying question.
- Ask ONE concise question at a time; be professional and brief.
"""

TECH_SYSTEM = f"""You are a senior DSA interviewer.

{COMMON_ANTI_HALLUCINATION}

Conduct a realistic, step-by-step technical interview with these strict rules:

CORE BEHAVIOR
- Start directly with a clear DSA question (no greetings/branding; do not mention any company).
- ALWAYS respond to the candidate's last answer first. Do not switch to a new problem until the current one is fully explored.
- Ask ONE concise follow-up at a time; never bundle multiple questions.
- Be professional, specific, and constructive (avoid generic praise like "great" without substance).

DEPTH-FIRST FOLLOW-UP (APPLY IN THIS ORDER)
For the SAME problem, iterate through this checklist:
1) Approach clarity: briefly acknowledge what they said and ask 1 targeted clarification if needed.
2) Correctness probe: ask about corner cases that could break their approach.
3) Complexity: ask for precise time and space complexity.
4) Optimization: ask if it can be improved (only after they've stated complexity).
5) Implementation: ask for brief pseudocode or the key steps (not full code) if appropriate.

Only after the checklist is satisfied and the solution is correct may you move difficulty upward. If the solution is incorrect:
- Give one short, subtle hint and ask them to retry on the SAME problem.
- If they fail twice, simplify the question (do not switch topics abruptly).

DIFFICULTY CONTROL (EMOTION-AWARE)
- If emotion suggests confidence/calm/concentration → increase difficulty ONLY after the checklist is completed.
- If emotion suggests nervous/doubt/confusion/frustration → keep/ease difficulty; give gentle hints; focus on clarity and one sub-step at a time.

SCOPE
- Focus purely on DSA: arrays, strings, linked lists, stacks/queues, trees, graphs, recursion/backtracking, DP, sorting/searching, hashing, two pointers, prefix/suffix, sliding window.
- Do not mention any company names or claim where the question is asked.

OUTPUT FORMAT
- Return only the NEXT single follow-up question (one sentence or two at most), referencing their last answer specifically.
- If information is missing or unclear, ask a brief, concrete clarifying question about their approach.
- Never say or repeat the word "undefined"; if the answer is unavailable, acknowledge briefly and ask a precise clarifying question.

DIFFICULTY PROGRESSION GUARD
- You may move from EASY → MEDIUM → HARD only after: (a) approach is correct, (b) complexity stated, (c) edge cases discussed, and (d) a brief implementation outline has been covered.
- Otherwise, stay on the SAME problem and continue the checklist or provide a hint.

Return only the NEXT question to ask the candidate, tailored to their last answer."""

PROJECT_SYSTEM = f"""You are a senior engineer for a project-based interview.
{COMMON_ANTI_HALLUCINATION}
- Keep project round chat separate from other rounds.
- Ask about projects: architecture, data modeling, trade-offs, performance, testing, deployments.
- Reference the candidate's actual projects and skills from their resume.
- If projects unknown, first ask for a brief outline of their top projects, then go deeper.
- ALWAYS acknowledge the last answer; one concise follow-up at a time.
- Emotion: confident → deeper implementation details; nervous → simplify + gentle hints.
Return only the NEXT question text."""

HR_SYSTEM = f"""You are an HR interviewer (behavioral).
{COMMON_ANTI_HALLUCINATION}
- Keep HR round chat separate from other rounds.
- Ask warm, professional questions: teamwork, conflict, leadership, achievements, motivation.
- Reference the candidate's actual experiences and achievements from their resume.
- ALWAYS acknowledge the last answer; one concise follow-up at a time.
- Emotion: confident → deeper probing; nervous → simpler + encouraging questions.
Return only the NEXT question text."""

ROUND_SYSTEM = {"technical": TECH_SYSTEM, "project": PROJECT_SYSTEM, "hr": HR_SYSTEM}

# -------- Helpers --------
def sanitize_text(t: Optional[str]) -> str:
    if not t: return "[no answer]"
    t = t.strip()
    if not t or t.lower() in ("undefined", "null"): return "[no answer]"
    return t

def emotion_hint(emotion: str) -> str:
    e = (emotion or "").lower()
    if any(k in e for k in ["confident", "calm", "concentrat"]):
        return "Candidate seems confident; increase difficulty slightly."
    if any(k in e for k in ["nervous", "doubt", "confus", "frustrat"]):
        return "Candidate seems nervous; simplify and offer a gentle hint."
    return "Emotion unclear; continue with balanced difficulty."

def difficulty_from_emotion(emotion: str) -> str:
    e = (emotion or "").lower()
    if any(k in e for k in ["confident", "calm", "concentrat"]): return "up"
    if any(k in e for k in ["nervous", "doubt", "confus", "frustrat"]): return "down"
    return "steady"

async def get_or_create_conversation(conversation_id: Optional[str], round_: RoundType) -> Dict[str, Any]:
    if conversation_id:
        doc = await chats.find_one({"conversation_id": conversation_id, "round": round_})
        if doc:
            logger.debug(f"[CONV] Reusing conversation_id={conversation_id} round={round_}")
            return doc
        logger.debug(f"[CONV] conversation_id={conversation_id} not found; creating new for round={round_}")
    conv_id = conversation_id or str(uuid.uuid4())
    new_doc = {
        "conversation_id": conv_id,
        "round": round_,
        "messages": [],
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    res = await chats.insert_one(new_doc)
    new_doc["_id"] = res.inserted_id
    logger.info(f"[CONV] Created conversation_id={conv_id} round={round_}")
    return new_doc

async def append_message(conv_id, role: str, content: str,
                         *, emotion: Optional[str]=None,
                         question_id: Optional[str]=None,
                         difficulty: Optional[str]=None):
    doc = {"role": role, "content": content, "ts": int(time.time())}
    if emotion is not None: doc["emotion"] = emotion
    if question_id is not None: doc["question_id"] = question_id
    if difficulty is not None: doc["difficulty"] = difficulty
    try:
        await chats.update_one(
            {"_id": conv_id},
            {"$push": {"messages": doc}, "$set": {"updated_at": int(time.time())}}
        )
        logger.debug(f"[MSG] Stored role={role} len={len(content)} emotion={emotion} diff={difficulty}")
    except Exception as e:
        logger.error(f"[MSG] Mongo write failed: {repr(e)}")
        raise

def to_history(messages: List[Dict[str,str]], max_turns: int = 20) -> List[Dict[str,str]]:
    trimmed = [m for m in messages if m.get("role") in ("user","assistant")][-max_turns:]
    return [{"role": m["role"], "content": m["content"]} for m in trimmed]

async def generate_next_question(system_prompt: str,
                                 history_msgs: List[Dict[str,str]],
                                 emotion: str,
                                 last_answer: Optional[str]) -> str:
    user_signal = f"""Emotion: {emotion}
Last answer: {sanitize_text(last_answer)}
{emotion_hint(emotion)}
Acknowledge the answer and ask ONE next question only."""
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history_msgs)
    msgs.append({"role": "user", "content": user_signal})

    logger.debug(f"[OPENAI] messages={len(msgs)} history={len(history_msgs)} emotion={emotion}")
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=msgs,
            temperature=0.3,
            top_p=0.2,
            max_tokens=300,
        )
        out = resp.choices[0].message.content.strip()
        logger.debug(f"[OPENAI] OK len={len(out)}")
        return out
    except Exception as e:
        logger.error(f"[OPENAI] ERROR: {repr(e)}")
        return "Let's proceed step by step. Could you outline your approach and its time/space complexity?"

async def generate_project_question(system_prompt: str,
                                   history_msgs: List[Dict[str,str]],
                                   emotion: str,
                                   last_answer: Optional[str],
                                   projects: List[str],
                                   skills: List[str]) -> str:
    user_signal = f"""Emotion: {emotion}
Last answer: {sanitize_text(last_answer)}
Projects: {', '.join(projects) if projects else 'No projects listed'}
Skills: {', '.join(skills) if skills else 'No skills listed'}
{emotion_hint(emotion)}
Acknowledge the answer and ask ONE next question only."""
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history_msgs)
    msgs.append({"role": "user", "content": user_signal})

    logger.debug(f"[OPENAI] messages={len(msgs)} history={len(history_msgs)} emotion={emotion} projects={len(projects)} skills={len(skills)}")
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=msgs,
            temperature=0.3,
            top_p=0.2,
            max_tokens=300,
        )
        out = resp.choices[0].message.content.strip()
        logger.debug(f"[OPENAI] OK len={len(out)}")
        return out
    except Exception as e:
        logger.error(f"[OPENAI] ERROR: {repr(e)}")
        return "Let's proceed step by step. Could you outline your approach and its time/space complexity?"

async def generate_hr_question(system_prompt: str,
                            history_msgs: List[Dict[str,str]],
                            emotion: str,
                            last_answer: Optional[str],
                            experiences: List[str],
                            achievements: List[str]) -> str:
    user_signal = f"""Emotion: {emotion}
Last answer: {sanitize_text(last_answer)}
Experiences: {', '.join(experiences) if experiences else 'No experiences listed'}
Achievements: {', '.join(achievements) if achievements else 'No achievements listed'}
{emotion_hint(emotion)}
Acknowledge the answer and ask ONE next question only."""
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history_msgs)
    msgs.append({"role": "user", "content": user_signal})

    logger.debug(f"[OPENAI] messages={len(msgs)} history={len(history_msgs)} emotion={emotion} experiences={len(experiences)} achievements={len(achievements)}")
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=msgs,
            temperature=0.3,
            top_p=0.2,
            max_tokens=300,
        )
        out = resp.choices[0].message.content.strip()
        logger.debug(f"[OPENAI] OK len={len(out)}")
        return out
    except Exception as e:
        logger.error(f"[OPENAI] ERROR: {repr(e)}")
        return "Let's proceed step by step. Could you outline your approach and its time/space complexity?"

async def handle_round(round_: RoundType,
                       req: AskRequest,
                       x_conversation_id: Optional[str]) -> AskResponse:
    logger.info(f"[REQ] round={round_} emotion={req.emotion} conv_header={x_conversation_id} ans_len={len((req.last_answer or '').strip())}")
    system_prompt = ROUND_SYSTEM[round_]
    conv = await get_or_create_conversation(x_conversation_id, round_)
    history = to_history(conv.get("messages", []), max_turns=20)
    logger.debug(f"[CTX] history_turns={len(history)}")

    await append_message(
        conv["_id"], "user",
        f"(answer) {sanitize_text(req.last_answer)}",
        emotion=req.emotion
    )

    next_q = await generate_next_question(system_prompt, history, req.emotion, req.last_answer)

    await append_message(conv["_id"], "assistant", next_q, difficulty=difficulty_from_emotion(req.emotion))

    logger.info(f"[RESP] round={round_} conv_id={conv['conversation_id']} q_len={len(next_q)}")
    return AskResponse(question=next_q, round=round_, conversation_id=conv["conversation_id"])

async def handle_project_round(req: ProjectAskRequest,
                              x_conversation_id: Optional[str]) -> AskResponse:
    logger.info(f"[REQ] project emotion={req.emotion} conv_header={x_conversation_id} ans_len={len((req.last_answer or '').strip())} projects={len(req.projects)} skills={len(req.skills)}")
    system_prompt = ROUND_SYSTEM["project"]
    conv = await get_or_create_conversation(x_conversation_id, "project")
    history = to_history(conv.get("messages", []), max_turns=20)
    logger.debug(f"[CTX] history_turns={len(history)}")

    await append_message(
        conv["_id"], "user",
        f"(answer) {sanitize_text(req.last_answer)}",
        emotion=req.emotion
    )

    next_q = await generate_project_question(system_prompt, history, req.emotion, req.last_answer, req.projects, req.skills)

    await append_message(conv["_id"], "assistant", next_q, difficulty=difficulty_from_emotion(req.emotion))

    logger.info(f"[RESP] project conv_id={conv['conversation_id']} q_len={len(next_q)}")
    return AskResponse(question=next_q, round="project", conversation_id=conv["conversation_id"])

async def handle_hr_round(req: HrAskRequest,
                        x_conversation_id: Optional[str]) -> AskResponse:
    logger.info(f"[REQ] hr emotion={req.emotion} conv_header={x_conversation_id} ans_len={len((req.last_answer or '').strip())} experiences={len(req.experiences)} achievements={len(req.achievements)}")
    system_prompt = ROUND_SYSTEM["hr"]
    conv = await get_or_create_conversation(x_conversation_id, "hr")
    history = to_history(conv.get("messages", []), max_turns=20)
    logger.debug(f"[CTX] history_turns={len(history)}")

    await append_message(
        conv["_id"], "user",
        f"(answer) {sanitize_text(req.last_answer)}",
        emotion=req.emotion
    )

    next_q = await generate_hr_question(system_prompt, history, req.emotion, req.last_answer, req.experiences, req.achievements)

    await append_message(conv["_id"], "assistant", next_q, difficulty=difficulty_from_emotion(req.emotion))

    logger.info(f"[RESP] hr conv_id={conv['conversation_id']} q_len={len(next_q)}")
    return AskResponse(question=next_q, round="hr", conversation_id=conv["conversation_id"])

# -------- Ask routes --------
@app.post("/api/technical", response_model=AskResponse)
async def technical(req: AskRequest, request: Request, x_conversation_id: Optional[str] = Header(None)):
    return await handle_round("technical", req, x_conversation_id)

@app.post("/api/project", response_model=AskResponse)
async def project(req: ProjectAskRequest, request: Request, x_conversation_id: Optional[str] = Header(None)):
    return await handle_project_round(req, x_conversation_id)

@app.post("/api/hr", response_model=AskResponse)
async def hr(req: HrAskRequest, request: Request, x_conversation_id: Optional[str] = Header(None)):
    return await handle_hr_round(req, x_conversation_id)

# -------- History routes --------
async def fetch_history(round_: RoundType, conversation_id: str, limit: int = 50) -> HistoryResponse:
    logger.info(f"[HISTORY] round={round_} conv_id={conversation_id} limit={limit}")
    doc = await chats.find_one({"conversation_id": conversation_id, "round": round_})
    if not doc:
        logger.warning("[HISTORY] conversation not found")
        raise HTTPException(status_code=404, detail="Conversation not found for this round.")
    msgs = doc.get("messages", [])
    trimmed = msgs[-limit:]
    return HistoryResponse(
        conversation_id=conversation_id,
        round=round_,
        messages=[HistoryMessage(**m) for m in trimmed]
    )

@app.get("/api/history/technical", response_model=HistoryResponse)
async def get_technical_history(
    x_conversation_id: Optional[str] = Header(None),
    conversation_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    conv_id = conversation_id or x_conversation_id
    if not conv_id:
        logger.error("[HISTORY] missing conversation_id")
        raise HTTPException(status_code=400, detail="conversation_id (query) or X-Conversation-Id (header) is required.")
    return await fetch_history("technical", conv_id, limit)

@app.get("/api/history/project", response_model=HistoryResponse)
async def get_project_history(
    x_conversation_id: Optional[str] = Header(None),
    conversation_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    conv_id = conversation_id or x_conversation_id
    if not conv_id:
        logger.error("[HISTORY] missing conversation_id")
        raise HTTPException(status_code=400, detail="conversation_id (query) or X-Conversation-Id (header) is required.")
    return await fetch_history("project", conv_id, limit)

@app.get("/api/history/hr", response_model=HistoryResponse)
async def get_hr_history(
    x_conversation_id: Optional[str] = Header(None),
    conversation_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    conv_id = conversation_id or x_conversation_id
    if not conv_id:
        logger.error("[HISTORY] missing conversation_id")
        raise HTTPException(status_code=400, detail="conversation_id (query) or X-Conversation-Id (header) is required.")
    return await fetch_history("hr", conv_id, limit)

@app.get("/health")
async def health():
    return {"status": "OK"}