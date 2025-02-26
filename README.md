# Crypto Sentiment Analysis with EMA Strategy

โปรเจคนี้เป็นแอปพลิเคชัน Streamlit สำหรับวิเคราะห์ความเชื่อมั่น (Sentiment Analysis) ของตลาด Crypto พร้อมกลยุทธ์การซื้อขายด้วย EMA (Exponential Moving Average)

## คุณสมบัติเด่น
- วิเคราะห์ Sentiment Analysis จากไฟล์ CSV
- คำนวณ EMA ระยะสั้นและระยะยาว พร้อมปรับแต่งได้ผ่าน UI
- ตรวจสอบสัญญาณซื้อ (Signal Buy) โดยอิงจากเกณฑ์ Sentiment และกลยุทธ์ EMA
- แสดงกราฟข้อมูลราคา, EMA, Sentiment และจุด Signal Buy ด้วย Plotly
- ประเมินผลตอบแทนในช่วงต่างๆ หลังเกิด Signal Buy

## วิธีการติดตั้งและรันโปรแกรม

### 1. Clone โครงการนี้
```bash
git clone https://github.com/AutolootDY/sentiment_BTC_ALL_DATA.git
cd <sentiment>
```

### 2. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install -r requirements.txt
```

ไฟล์ `requirements.txt` ควรมี:
```
pandas
plotly
streamlit
```

### 3. รันแอปพลิเคชัน
```bash
streamlit run app.py
```

## โครงสร้างไฟล์
```
.
├── app.py                 # โค้ดหลักของแอป
├── mt5_data_XAUUSD_TF1H_FUB_TH.csv  # ข้อมูลราคาทองคำ
├── daily_sentiment.csv    # ข้อมูล Sentiment รายวัน
└── requirements.txt       # ไลบรารีที่ต้องติดตั้ง
```

## วิธีการใช้งาน
1. อัปโหลดไฟล์ CSV ของราคาและ Sentiment ผ่าน Sidebar ของแอป
2. ปรับค่าระยะ EMA และจำนวนวันที่ต้องการตรวจสอบสัญญาณ
3. กดปุ่ม "Run Analysis" เพื่อเริ่มกระบวนการวิเคราะห์
4. ดูข้อมูลที่รวมกัน (Merged Data), กราฟการซื้อขาย และกราฟผลตอบแทน

## ผลลัพธ์
- DataFrame ที่รวมข้อมูลราคากับ Sentiment
- กราฟแสดงสัญญาณซื้อและราคา
- กราฟแสดงผลตอบแทนในช่วง 1, 3, 5, 10, 15, และ 30 วัน

🔗 [ดูกราฟวิเคราะห์ตลาดได้ที่นี่](https://sentimentbtcalldata-hz8scocudczrvkx3mvkcmb.streamlit.app/) 🚀📊



## หมายเหตุ
- สัญญาณซื้อเกิดขึ้นเมื่อ Sentiment ต่ำกว่า -30 และกลับมาเป็นบวก พร้อมกับ EMA ระยะสั้นตัด EMA ระยะยาวจากล่างขึ้นบน
- ข้อมูลที่ไม่มีค่า (NaN) จะถูกแทนที่ด้วย 0 โดยอัตโนมัติ

## ติดต่อ
หากมีคำถามหรือข้อสงสัย สามารถติดต่อได้ที่ [Your Contact Information]

