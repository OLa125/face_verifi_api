# Face Verification API

API للتحقق من تطابق وجهين باستخدام insightface.

## طريقة التشغيل

```bash
pip install -r requirements.txt
python app.py
```

### إرسال طلب POST

ارفع صورتين كـ form-data مع المفاتيح: `image1` و `image2` إلى:

```
http://localhost:5000/verify
```

### الاستجابة

```json
{
  "similarity": 0.83,
  "same_person": true
}
```