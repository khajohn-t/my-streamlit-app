import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time # For exponential backoff

# 1. Imports and Configuration
try:
    # Use the Google GenAI SDK (pip install google-genai)
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    st.error("ไลบรารี 'google-genai' ไม่ได้ถูกติดตั้ง กรุณาติดตั้งโดยใช้: pip install google-genai")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"

# 2. Utility Functions

def get_article_text(url):
    """ดึงข้อความหลักจาก URL ข่าว."""
    try:
        # Set a common User-Agent to avoid being blocked by some websites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to find main text from common article tags (p, h1-h3)
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        
        # Filter empty text and join them
        article_text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        # If text is too short, try using the whole body as fallback
        if len(article_text) < 100:
             article_text = soup.body.get_text(separator='\n', strip=True)
             
        if not article_text:
            return None, "ไม่พบเนื้อหาที่ชัดเจนในหน้านี้"

        # Limit the text size sent to the LLM to save cost and prevent exceeding limits
        # Note: This is the raw (noisy) text limit. The cleaned text will be shorter.
        max_length = 15000
        if len(article_text) > max_length:
            article_text = article_text[:max_length] + "..."
            st.warning(f"ข้อความข่าวที่ดึงมา (ก่อนการกรอง) ถูกตัดให้เหลือเพียง {max_length} ตัวอักษรเพื่อความรวดเร็วในการประมวลผล")

        return article_text, None

    except requests.exceptions.RequestException as e:
        return None, f"ไม่สามารถดึงข้อมูลจาก URL ได้: {e}"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการประมวลผล: {e}"


def extract_main_content_with_gemini(client, noisy_text):
    """ใช้ Gemini เพื่อดึงเฉพาะเนื้อหาหลักของบทความจากข้อความที่อาจมีสิ่งรบกวน."""
    extraction_prompt = f"""
    คุณคือผู้ช่วยดึงเนื้อหาหลัก (Core Article Extractor)
    จงวิเคราะห์ข้อความต่อไปนี้ซึ่งถูกดึงมาจากหน้าเว็บข่าว
    ข้อความนี้อาจมีเนื้อหาที่ไม่เกี่ยวข้อง เช่น เมนูนำทาง, โฆษณา, คำบรรยายรูปภาพ, หรือส่วนท้ายของเว็บไซต์
    หน้าที่ของคุณคือ:
    1.  คัดเลือก **เฉพาะเนื้อหาหลักของบทความข่าว** (บทนำ, ย่อหน้าเนื้อหา, บทสรุป)
    2.  ละเว้นส่วนที่ไม่ใช่เนื้อหาหลัก เช่น ส่วนหัว, ส่วนท้าย, เมนู, ลิงก์ที่เกี่ยวข้อง, และคำอธิบายภาพที่ไม่ใช่เนื้อหา
    3.  ตอบกลับด้วยเนื้อหาหลักที่ถูกคัดเลือกมาเท่านั้น

    --- ข้อความที่ถูกดึงมา ---
    {noisy_text}
    """
    
    # Configure the API call for text extraction
    extraction_system_instruction = "คุณคือ Core Article Extractor ที่แม่นยำและตอบกลับด้วยข้อความที่สะอาดเท่านั้น"
    extraction_config = types.GenerateContentConfig(
        system_instruction=extraction_system_instruction
    )

    response = make_gemini_call_with_retry(
        client,
        contents=[extraction_prompt],
        config=extraction_config
    )

    if response and response.text:
        return response.text.strip(), None
    else:
        return None, "ไม่สามารถดึงเนื้อหาหลักด้วย Gemini ได้"


def make_gemini_call_with_retry(client, contents, config=None, max_retries=3):
    """เรียกใช้ Gemini API พร้อมกลไก Exponential Backoff."""
    for attempt in range(max_retries):
        try:
            # Call the API with contents and configuration
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            return response
        except APIError as e:
            if attempt < max_retries - 1:
                # Exponential backoff logic: 1s, 2s, 4s wait times
                wait_time = 2 ** attempt 
                st.warning(f"เกิดข้อผิดพลาดจาก API ({e}) ลองใหม่ใน {wait_time} วินาที...")
                time.sleep(wait_time)
            else:
                st.error(f"การเรียก API ล้มเหลวหลังจาก {max_retries} ครั้ง: {e}")
                return None
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดในการเรียก API: {e}")
            return None
    return None

# 3. Streamlit App Layout and Logic

st.set_page_config(
    page_title="เรียนภาษาอังกฤษจากข่าว",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 เรียนภาษาอังกฤษจากข่าว")
st.markdown("ใส่ URL ข่าวภาษาอังกฤษ เพื่อรับการสรุปภาษาไทยและตารางคำศัพท์สำหรับฝึกฝน!")

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("การตั้งค่า API")
    gemini_api_key = st.text_input(
        "กรุณาใส่ Gemini API Key ของคุณ",
        type="password",
        key="gemini_api_key"
    )
    st.markdown("หากยังไม่มีคีย์ สามารถรับได้จาก [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)")

# --- Main Input ---
news_url = st.text_input(
    "ใส่ URL ของบทความข่าวภาษาอังกฤษที่นี่:",
    key="news_url",
    placeholder="เช่น https://www.bbc.com/news/world-us-canada-67616140"
)

# Create the process button
process_button = st.button("ประมวลผลข่าว", type="primary", key="process_news_button")

if process_button:
    # 1. Input Validation
    if not news_url:
        st.info("กรุณาใส่ URL ของบทความข่าว")
        st.stop()
        
    if not gemini_api_key:
        st.error("กรุณาใส่ Gemini API Key ในแถบด้านข้างก่อน!")
        st.stop()
    
    if not news_url.startswith(('http://', 'https://')):
        st.error("URL ไม่ถูกต้อง กรุณาใส่ URL ที่ขึ้นต้นด้วย http:// หรือ https://")
        st.stop()

    # --- Step 1: Extract Article Text (Noisy) ---
    with st.spinner("กำลังดึงข้อความจาก URL..."):
        noisy_article_text, error = get_article_text(news_url)

    if error:
        st.error(error)
        st.stop()

    if not noisy_article_text or len(noisy_article_text) < 50:
        st.error("ไม่สามารถดึงข้อความข่าวที่มีความหมายได้ กรุณาลอง URL อื่น.")
        st.stop()

    st.success("ดึงข้อความข่าวสำเร็จ!")

    try:
        # Initialise Gemini Client
        client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        st.error(f"ไม่สามารถเริ่มต้น Gemini Client ได้: {e}. ตรวจสอบ API Key ของคุณ.")
        st.stop()
        
    # --- Step 1.5: Clean Article Text with Gemini ---
    st.subheader("1.5 การกรองเนื้อหาหลักอัตโนมัติ")
    with st.spinner("กำลังให้ Gemini กรองเฉพาะเนื้อหาข่าวหลัก..."):
        clean_article_text, extraction_error = extract_main_content_with_gemini(client, noisy_article_text)
        
    if extraction_error:
        st.error(extraction_error)
        # Fallback: Use the original noisy text if cleaning fails
        clean_article_text = noisy_article_text 
        st.warning("เนื่องจากเกิดข้อผิดพลาดในการกรองเนื้อหาหลัก ระบบจะใช้ข้อความที่ดึงมาทั้งหมดแทน (อาจมีสิ่งรบกวน)")
    else:
        st.success("กรองเนื้อหาหลักสำเร็จ!")

    # --- Display Part 1: Cleaned Text ---
    st.header("1. ข้อความข่าวภาษาอังกฤษฉบับหลักที่ถูกกรองแล้ว")
    st.text_area(
        "เนื้อหาข่าวที่ถูกกรองแล้ว:", 
        clean_article_text, 
        height=300, 
        disabled=True,
        key="cleaned_text"
    )

    # --- Step 2: Generate Thai Summary (Part 2) ---
    st.header("2. สรุปข่าวเป็นภาษาไทย")
    with st.spinner("กำลังให้ Gemini สรุปข่าวเป็นภาษาไทย..."):
        # Use the CLEANED text for the prompt
        summary_prompt = f"สรุปเนื้อหาข่าวภาษาอังกฤษต่อไปนี้ให้เป็นภาษาไทยที่กระชับและเข้าใจง่าย ในรูปแบบย่อหน้าเดียว:\n\n---\n\n{clean_article_text}"
        summary_system_instruction = "คุณคือผู้ช่วยสรุปข่าวที่เชี่ยวชาญภาษาไทย"

        # Create config object and pass system_instruction inside
        summary_config = types.GenerateContentConfig(
            system_instruction=summary_system_instruction
        )

        summary_response = make_gemini_call_with_retry(
            client, 
            contents=[summary_prompt], 
            config=summary_config
        )
        
        if summary_response and summary_response.text:
            st.markdown(f"**สรุป:** {summary_response.text}")
        else:
            st.error("ไม่สามารถสร้างบทสรุปได้ (โปรดตรวจสอบ API Key และโควต้า)")


    # --- Step 3: Generate Vocabulary Table (Part 3) ---
    st.header("3. ตารางคำศัพท์และตัวอย่างประโยค")
    with st.spinner("กำลังให้ Gemini สร้างตารางคำศัพท์ 5 คำ..."):
        
        # Define the JSON Schema for structured output
        vocab_schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "English_Word": types.Schema(type=types.Type.STRING, description="ศัพท์ภาษาอังกฤษระดับมัธยมจากข่าว"),
                    "Thai_Translation": types.Schema(type=types.Type.STRING, description="คำแปลภาษาไทย"),
                    "Example_Sentence": types.Schema(type=types.Type.STRING, description="ประโยคเต็มที่ใช้คำนั้นจากข้อความข่าวเดิม")
                },
                required=["English_Word", "Thai_Translation", "Example_Sentence"]
            )
        )
        
        vocab_system_instruction = "คุณคือครูสอนภาษาอังกฤษที่เชี่ยวชาญการสร้างบทเรียนจากเนื้อหาจริง คุณต้องตอบกลับเป็น JSON ที่ตรงตาม Schema ที่กำหนดเท่านั้น"

        # Create config object and pass system_instruction and schema inside
        vocab_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=vocab_schema,
            system_instruction=vocab_system_instruction
        )
        
        # Use the CLEANED text for the prompt
        vocab_prompt = f"จากข้อความข่าวต่อไปนี้ ให้คุณสร้างรายการคำศัพท์ 5 คำที่เหมาะสำหรับนักเรียนระดับมัธยมปลาย พร้อมคำแปลภาษาไทย และตัวอย่างประโยคที่ใช้คำนั้น ซึ่งต้องมาจากข้อความข่าวเดิมเท่านั้น:\n\n---\n\n{clean_article_text}"

        vocab_response = make_gemini_call_with_retry(
            client, 
            contents=[vocab_prompt], 
            config=vocab_config
        )

        if vocab_response and vocab_response.text:
            try:
                # Parse the JSON string output
                vocab_data = json.loads(vocab_response.text)
                
                # Convert to DataFrame and rename columns for display
                vocab_df = pd.DataFrame(vocab_data)
                vocab_df.columns = ["ศัพท์ภาษาอังกฤษ", "คำแปลภาษาไทย", "ตัวอย่างประโยค (จากข่าว)"]

                # Display the DataFrame
                st.dataframe(
                    vocab_df, 
                    use_container_width=True, 
                    hide_index=True,
                    # Set column width to give more space for example sentences
                    column_config={
                        "ตัวอย่างประโยค (จากข่าว)": st.column_config.TextColumn(
                            "ตัวอย่างประโยค (จากข่าว)",
                            width="large"
                        )
                    }
                )

            except json.JSONDecodeError:
                st.error("ข้อผิดพลาด: Gemini ตอบกลับเป็นรูปแบบ JSON ที่ไม่ถูกต้อง")
                st.text(vocab_response.text)
            except Exception as e:
                st.error(f"ข้อผิดพลาดในการแสดงผลตาราง: {e}")
        else:
            st.error("ไม่สามารถสร้างตารางคำศัพท์ได้ (โปรดตรวจสอบ API Key และโควต้า)")