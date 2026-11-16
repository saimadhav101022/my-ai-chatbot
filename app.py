
import os
import json
import uuid
import zlib
import base64
import time

from datetime import datetime, date, timedelta
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CAM_VISION",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 7rem;
}

section[data-testid="stSidebar"] {
    width: 290px !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.cam-header {
    border-bottom: 1px solid rgba(128,128,128,0.25);
    padding: 10px 0 18px 0;
    margin-bottom: 20px;
}

.cam-title {
    font-size: 1.5rem;
    font-weight: 700;
}

.cam-subtitle {
    font-size: 0.85rem;
    opacity: 0.65;
}

.welcome {
    text-align: center;
    margin-top: 15vh;
}

.welcome-icon {
    font-size: 3rem;
}

.welcome-title {
    font-size: 2rem;
    font-weight: 700;
}

.welcome-text {
    opacity: 0.65;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY not found. "
        "Please check your .env file."
    )
    st.stop()


# ============================================================
# 4. GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

try:
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            timeout=60000
        )
    )

except Exception as e:
    st.error("Gemini client initialization failed.")
    st.exception(e)
    st.stop()


# ============================================================
# 5. LOAD RESTAURANT KNOWLEDGE
# ============================================================

KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "restaurant_knowledge.json"
)


def load_restaurant_knowledge():

    try:

        with open(
            KNOWLEDGE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            knowledge = json.load(file)

        if not isinstance(knowledge, dict):
            raise ValueError("JSON must contain an object.")

        return knowledge

    except FileNotFoundError:

        st.error(
            "restaurant_knowledge.json was not found. "
            "Place it in the same folder as app.py."
        )

        return {}

    except Exception as e:

        st.error("Restaurant knowledge loading failed.")
        st.exception(e)

        return {}


RESTAURANT_KB = load_restaurant_knowledge()


# ============================================================
# 6. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTION = """
You are CAM_VISION, an intelligent AI chatbot.

YOUR IDENTITY:
Your name is CAM_VISION.
You are developed using Python and Streamlit.
You use Google's Gemini API.

GENERAL PURPOSE:
Answer general questions.
Explain programming and technical concepts.
Help with electronics, embedded systems and IoT.
Solve mathematical problems.
Have natural conversations.

RESPONSE STYLE:
Be friendly and professional.
Give clear and accurate answers.
Use simple explanations when appropriate.
Give detailed answers when requested.

RESTAURANT RULES:
When a customer asks about Aroma Restaurant, use the
restaurant knowledge supplied in the conversation.

Never invent restaurant menu items or prices.
Never invent restaurant timings or services.
Use the exact prices from the knowledge base.
Mention half and full prices separately.
Mention family packs and jumbo packs separately.
If information is missing, say it is not available.
If a service is unconfirmed, do not claim it is available.
Respond in English.

For restaurant contact enquiries, use:
+91 85198 15775

For questions unrelated to the restaurant,
answer normally.
"""


# ============================================================
# 7. RESTAURANT QUESTION DETECTION
# ============================================================

RESTAURANT_KEYWORDS = [
    "aroma",
    "restaurant",
    "menu",
    "biryani",
    "biriyani",
    "starter",
    "soup",
    "noodles",
    "paneer",
    "chicken",
    "mutton",
    "prawns",
    "fish",
    "seafood",
    "roti",
    "naan",
    "parotta",
    "dessert",
    "ice cream",
    "kulfi",
    "beverage",
    "thums",
    "sprite",
    "family pack",
    "jumbo pack",
    "food price",
    "food items",
    "home delivery",
    "takeaway",
    "take away",
    "table booking",
    "opening time",
    "closing time",
    "restaurant address",
    "restaurant phone",
    "restaurant contact",
    "restaurant timing"
]


def normalize_text(text):

    return " ".join(
        str(text).lower()
        .replace("&", "and")
        .replace("-", " ")
        .split()
    )


def is_restaurant_question(question):

    question = normalize_text(question)

    return any(
        keyword in question
        for keyword in RESTAURANT_KEYWORDS
    )


# ============================================================
# 8. RESTAURANT KNOWLEDGE RETRIEVAL
# ============================================================

def retrieve_restaurant_context(question, knowledge):

    if not knowledge:
        return None

    if not is_restaurant_question(question):
        return None

    q = normalize_text(question)

    results = []

    # Restaurant information

    results.append({
        "section": "restaurant",
        "information": knowledge.get("restaurant", {})
    })

    # Search menu and pack sections

    sections = [
        "menu",
        "family_packs",
        "jumbo_packs",
        "faqs",
        "policies",
        "special_information"
    ]

    # Categories and aliases

    category_aliases = {
        "soups": ["soup"],
        "noodles": ["noodle"],
        "veg starters": ["veg starter"],
        "chicken starters": ["chicken starter"],
        "mutton starters": ["mutton starter"],
        "sea foods": ["seafood", "fish", "prawn"],
        "chicken main course": ["chicken main course"],
        "mutton main course": ["mutton main course"],
        "egg main course": ["egg main course"],
        "biryani": ["biryani", "biriyani"],
        "beverages": ["beverage", "drink", "sprite", "thums"],
        "desserts": ["dessert", "sweet", "ice cream", "kulfi"],
        "roti": ["roti", "naan", "parotta"],
        "ice creams": ["ice cream", "scoop"],
        "fruit salads": ["fruit salad"],
        "cones": ["cone"]
    }

    requested_category = None

    for category, aliases in category_aliases.items():

        if category in q or any(
            alias in q for alias in aliases
        ):

            requested_category = category
            break

    ignored_words = {
        "what", "is", "the", "price", "of", "how",
        "much", "do", "you", "have", "for", "please",
        "tell", "me", "a", "an", "and", "in", "at",
        "your", "available", "show", "list", "all",
        "items", "can", "i", "get", "give", "are",
        "there", "any", "with", "does", "restaurant"
    }

    for section in sections:

        records = knowledge.get(section, [])

        if not isinstance(records, list):
            continue

        for item in records:

            if not isinstance(item, dict):
                continue

            name = normalize_text(
                item.get("name", "")
            )

            category = normalize_text(
                item.get("category", "")
            )

            searchable = normalize_text(
                json.dumps(item, ensure_ascii=False)
            )

            score = 0

            # Exact item match

            if name and name in q:
                score += 20

            # Match words in the question

            for word in q.split():

                if (
                    len(word) > 2
                    and word not in ignored_words
                    and word in searchable
                ):
                    score += 2

            # Category match

            if requested_category:

                aliases = category_aliases.get(
                    requested_category,
                    []
                )

                if (
                    requested_category in category
                    or any(
                        alias in category
                        for alias in aliases
                    )
                ):
                    score += 20

            if score > 0:

                results.append({
                    "section": section,
                    "information": item,
                    "_score": score
                })

    # Include service information

    services = knowledge.get("services", {})

    if isinstance(services, dict):

        results.append({
            "section": "services",
            "information": services
        })

    # Sort results by relevance

    results.sort(
        key=lambda x: x.get("_score", 0),
        reverse=True
    )

    # Remove duplicates

    unique_results = []
    seen = set()

    for item in results:

        info = item["information"]

        key = json.dumps(
            info,
            sort_keys=True,
            ensure_ascii=False
        )

        if key not in seen:

            seen.add(key)

            unique_results.append({
                "section": item["section"],
                "information": info
            })

    if len(unique_results) == 1:

        unique_results.append({
            "section": "notice",
            "information": (
                "The requested information was not "
                "found in the restaurant knowledge base."
            )
        })

    context = {
        "restaurant": knowledge.get("restaurant", {}),
        "relevant_information": unique_results[:50]
    }

    return json.dumps(
        context,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 9. GEMINI RESPONSE GENERATION
# ============================================================

def generate_gemini_response(
    conversation,
    restaurant_context=None
):

    system_prompt = SYSTEM_INSTRUCTION

    if restaurant_context:

        system_prompt += """

RESTAURANT KNOWLEDGE BASE:

""" + restaurant_context + """

IMPORTANT RESTAURANT INSTRUCTIONS:

1. Use the supplied information as the source of truth.
2. Never invent prices or menu items.
3. Give exact half and full prices when available.
4. If the requested item is not listed, say so.
5. If a price is missing, say it needs confirmation.
6. Do not assume unconfirmed services are available.
7. Answer the customer's question directly.
"""

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=conversation,

            config=types.GenerateContentConfig(

                system_instruction=system_prompt,

                temperature=0.3

            )
        )

        if response.text:

            return response.text

        raise ValueError(
            "Gemini returned an empty response."
        )

    except Exception as e:

        print("\n========== GEMINI ERROR ==========")
        print(repr(e))
        print("==================================\n")

        raise


# ============================================================
# 10. CHAT MANAGEMENT
# ============================================================

def make_chat():

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    return {
        "title": "New Chat",
        "messages": [],
        "created_at": now,
        "updated_at": now
    }


def ensure_state():

    if "chats" not in st.session_state:

        chat_id = str(uuid.uuid4())

        st.session_state.chats = {
            chat_id: make_chat()
        }

        st.session_state.current_chat = chat_id

    if "current_chat" not in st.session_state:

        st.session_state.current_chat = next(
            iter(st.session_state.chats)
        )


def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = make_chat()

    st.session_state.current_chat = chat_id


def delete_chat(chat_id):

    if chat_id in st.session_state.chats:

        del st.session_state.chats[chat_id]

    if not st.session_state.chats:

        create_new_chat()

    elif st.session_state.current_chat == chat_id:

        st.session_state.current_chat = next(
            iter(st.session_state.chats)
        )


def generate_title(question):

    title = " ".join(question.strip().split())

    if len(title) > 38:

        title = title[:38] + "..."

    return title


def format_day(timestamp):

    try:

        dt = datetime.fromisoformat(
            timestamp
        ).date()

        today = date.today()

        if dt == today:
            return "Today"

        if dt == today - timedelta(days=1):
            return "Yesterday"

        return dt.strftime("%d %b %Y")

    except Exception:

        return "Older"


# ============================================================
# 11. SHARE CONVERSATIONS
# ============================================================

def encode_chat(chat_id):

    chat = st.session_state.chats[chat_id]

    payload = {
        "title": chat["title"],
        "messages": chat["messages"],
        "created_at": chat["created_at"],
        "updated_at": chat["updated_at"]
    }

    raw = json.dumps(payload).encode("utf-8")

    compressed = zlib.compress(raw, level=9)

    return base64.urlsafe_b64encode(
        compressed
    ).decode("ascii")


def decode_chat(token):

    try:

        compressed = base64.urlsafe_b64decode(
            token.encode("ascii")
        )

        raw = zlib.decompress(compressed)

        payload = json.loads(raw.decode("utf-8"))

        if not isinstance(payload, dict):
            return None

        if not isinstance(payload.get("messages"), list):
            return None

        for message in payload["messages"]:

            if (
                not isinstance(message, dict)
                or message.get("role") not in [
                    "user", "assistant"
                ]
                or not isinstance(
                    message.get("content"), str
                )
            ):
                return None

        return payload

    except Exception:

        return None


# ============================================================
# 12. INITIALIZE SESSION
# ============================================================

ensure_state()

if "shared_loaded" not in st.session_state:
    st.session_state.shared_loaded = False

if "share_chat_id" not in st.session_state:
    st.session_state.share_chat_id = None


# ============================================================
# 13. LOAD SHARED CONVERSATION
# ============================================================

share_token = st.query_params.get("share")

if share_token and not st.session_state.shared_loaded:

    shared = decode_chat(share_token)

    if shared:

        shared_id = str(uuid.uuid4())

        st.session_state.chats[shared_id] = {

            "title": "Shared: " + shared.get(
                "title", "Chat"
            ),

            "messages": shared.get("messages", []),

            "created_at": shared.get(
                "created_at",
                datetime.now().isoformat()
            ),

            "updated_at": shared.get(
                "updated_at",
                datetime.now().isoformat()
            )
        }

        st.session_state.current_chat = shared_id

        st.session_state.shared_loaded = True

    else:

        st.warning("Invalid shared conversation link.")


# ============================================================
# 14. CURRENT CHAT
# ============================================================

current_chat_id = st.session_state.current_chat

current_chat = st.session_state.chats[current_chat_id]


# ============================================================
# 15. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 CAM_VISION")

    st.caption("Python-Based AI Chatbot")

    st.markdown("---")

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary"
    ):

        create_new_chat()

        st.session_state.shared_loaded = False
        st.session_state.share_chat_id = None

        st.query_params.clear()

        st.rerun()

    search_text = st.text_input(
        "Search conversations",
        placeholder="Search conversations..."
    )

    st.markdown("---")

    all_chats = list(
        st.session_state.chats.items()
    )

    if search_text:

        search_lower = search_text.lower()

        all_chats = [
            item for item in all_chats
            if (
                search_lower in item[1]["title"].lower()
                or any(
                    search_lower in message["content"].lower()
                    for message in item[1]["messages"]
                )
            )
        ]

    all_chats.sort(
        key=lambda item: item[1]["updated_at"],
        reverse=True
    )

    grouped = {}

    for chat_id, chat in all_chats:

        day = format_day(chat["updated_at"])

        grouped.setdefault(day, []).append(
            (chat_id, chat)
        )

    for day, chats in grouped.items():

        st.markdown(f"**{day}**")

        for chat_id, chat in chats:

            col1, col2 = st.columns(
                [0.82, 0.18]
            )

            with col1:

                title = chat["title"][:31]

                if chat_id == st.session_state.current_chat:
                    title = "● " + title

                if st.button(
                    title,
                    key=f"open_{chat_id}",
                    use_container_width=True
                ):

                    st.session_state.current_chat = chat_id
                    st.rerun()

            with col2:

                with st.popover("⋮"):

                    new_title = st.text_input(
                        "Rename chat",
                        value=chat["title"],
                        key=f"rename_{chat_id}"
                    )

                    if st.button(
                        "Rename",
                        key=f"rename_btn_{chat_id}"
                    ):

                        if new_title.strip():

                            chat["title"] = new_title.strip()

                        st.rerun()

                    if st.button(
                        "Share",
                        key=f"share_{chat_id}"
                    ):

                        st.session_state.share_chat_id = chat_id
                        st.rerun()

                    chat_text = "\n\n".join(
                        [
                            (
                                "User: "
                                if message["role"] == "user"
                                else "CAM_VISION: "
                            ) + message["content"]
                            for message in chat["messages"]
                        ]
                    )

                    st.download_button(
                        "Export conversation",
                        data=chat_text or "Empty conversation",
                        file_name="CAM_VISION_chat.txt",
                        mime="text/plain",
                        key=f"export_{chat_id}"
                    )

                    if st.button(
                        "Delete",
                        key=f"delete_{chat_id}"
                    ):

                        delete_chat(chat_id)
                        st.rerun()

    st.markdown("---")

    with st.expander("About CAM_VISION"):

        st.write(
            "CAM_VISION is an AI-powered chatbot "
            "built using Python, Streamlit and Gemini."
        )

    with st.expander("Settings"):

        st.write("Model:", MODEL_NAME)

    st.markdown("---")

    st.caption(
        "Restaurant Knowledge: "
        + ("Loaded" if RESTAURANT_KB else "Not available")
    )


# ============================================================
# 16. SHARE PANEL
# ============================================================

share_chat_id = st.session_state.share_chat_id

if share_chat_id in st.session_state.chats:

    token = encode_chat(share_chat_id)

    st.info("Share this conversation")

    share_url = (
        "http://localhost:8501/?share="
        + quote(token)
    )

    st.code(share_url)

    st.caption(
        "For sharing with other users, deploy "
        "CAM_VISION online."
    )

    if st.button("Close Share Panel"):

        st.session_state.share_chat_id = None
        st.rerun()


# ============================================================
# 17. MAIN HEADER
# ============================================================

header_left, header_right = st.columns([0.8, 0.2])

with header_left:

    st.markdown("""
    <div class="cam-header">
        <div class="cam-title">
            🤖 CAM_VISION
        </div>
        <div class="cam-subtitle">
            Python-Based AI Chatbot
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_right:

    if st.button("🔗 Share", use_container_width=True):

        st.session_state.share_chat_id = current_chat_id
        st.rerun()


# ============================================================
# 18. WELCOME SCREEN
# ============================================================

if not current_chat["messages"]:

    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🤖</div>
        <div class="welcome-title">
            How can I help you today?
        </div>
        <div class="welcome-text">
            Ask CAM_VISION anything.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 19. DISPLAY CHAT HISTORY
# ============================================================

for message in current_chat["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            st.caption("CAM_VISION")


# ============================================================
# 20. USER INPUT
# ============================================================

question = st.chat_input(
    "Ask CAM_VISION anything..."
)


# ============================================================
# 21. PROCESS USER QUESTION
# ============================================================

if question:

    current_chat["messages"].append({
        "role": "user",
        "content": question
    })

    current_chat["updated_at"] = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    if len(current_chat["messages"]) == 1:

        current_chat["title"] = generate_title(question)

    with st.chat_message("user"):

        st.markdown(question)

    # Build conversation history.

    conversation = []

    for message in current_chat["messages"]:

        role = (
            "user"
            if message["role"] == "user"
            else "model"
        )

        conversation.append({
            "role": role,
            "parts": [
                {"text": message["content"]}
            ]
        })

    # Retrieve restaurant information.

    restaurant_context = retrieve_restaurant_context(
        question,
        RESTAURANT_KB
    )

    # Generate answer.

    with st.chat_message("assistant"):

        try:

            with st.spinner("CAM_VISION is thinking..."):

                answer = generate_gemini_response(
                    conversation,
                    restaurant_context
                )

            st.markdown(answer)

            st.caption("CAM_VISION")

        except Exception as e:

            # PRINT COMPLETE ERROR IN TERMINAL

            print("\n")
            print("=" * 60)
            print("CAM_VISION GEMINI ERROR")
            print("=" * 60)
            print("Error type:", type(e).__name__)
            print("Error details:", repr(e))
            print("=" * 60)
            print("\n")

            # SHOW COMPLETE ERROR IN STREAMLIT

            st.error(
                "CAM_VISION encountered an error. "
                "Here are the actual error details:"
            )

            st.exception(e)

            answer = (
                "CAM_VISION could not generate a response. "
                "Please check the error details displayed above."
            )

    # Save assistant response.

    current_chat["messages"].append({
        "role": "assistant",
        "content": answer
    })

    current_chat["updated_at"] = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    st.rerun()
