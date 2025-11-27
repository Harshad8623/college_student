from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Email, Length
import cv2
import numpy as np
import pymysql
import os
import pandas as pd
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlite3
import mysql.connector
import seaborn as sns  # optional
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import io
from PIL import Image
import json
import pymysql
from pymysql import IntegrityError
from datetime import datetime, date

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'training_images'

# --------------------------------------------------------------------
# MYSQL CONFIG
# --------------------------------------------------------------------
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_DB = "attendance_db"

# --------------------------------------------------------------------
# CONSTANTS / PATHS
# --------------------------------------------------------------------
DATA_DIR = 'data'          # still used for exports/other data if needed
MODEL_DIR = 'models'
TRAINING_DIR = 'training_images'
MODEL_PATH = os.path.join(MODEL_DIR, 'face_model.yml')
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.npy')

# smaller = better; we accept if confidence <= threshold
ATTENDANCE_CONFIDENCE_THRESHOLD = 85.0

# --------------------------------------------------------------------
# DB CONNECTION HELPERS
# --------------------------------------------------------------------
def get_connection(db_with_schema=True):
    """
    Returns a pymysql connection.
    If db_with_schema=False, connects without specifying DB (used to create DB).
    """
    if db_with_schema:
        return pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            autocommit=False
        )
    else:
        return pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            autocommit=True
        )

# --------------------------------------------------------------------
# DB INITIALIZATION
# --------------------------------------------------------------------
def init_db():
    # 1) Create database if not exists
    conn = get_connection(db_with_schema=False)
    c = conn.cursor()
    c.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` DEFAULT CHARACTER SET utf8mb4")
    c.close()
    conn.close()

    # 2) Create tables
    conn = get_connection()
    c = conn.cursor()

    # students table
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            class VARCHAR(50) NOT NULL,
            roll_number VARCHAR(50) UNIQUE NOT NULL,
            face_encoding LONGBLOB,
            approved TINYINT(1) DEFAULT 0,
            password VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB
    """)

    # teachers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB
    """)

    # attendance table with subject + unique constraint to prevent duplicates
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT,
            date DATE NOT NULL,
            subject VARCHAR(100) DEFAULT NULL,
            status ENUM('Present','Absent','Late') NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE KEY uniq_attendance (student_id, date, subject)
        ) ENGINE=InnoDB
    """)

    # marks table
    c.execute("""
        CREATE TABLE IF NOT EXISTS marks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT,
            subject VARCHAR(100) NOT NULL,
            marks INT NOT NULL,
            max_marks INT NOT NULL,
            exam_date DATE NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
                ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB
    """)
   # CREATE TABLE attendance
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
             (id INT PRIMARY KEY AUTO_INCREMENT
,
              student_id INTEGER,
              date DATE NOT NULL,
              subject TEXT,
              status TEXT NOT NULL,
              timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY (student_id) REFERENCES students (id))''')


    conn.commit()
    c.close()
    conn.close()



def migrate_db():
    """Ensure attendance table has 'subject' column (MySQL version)."""
    conn = pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        autocommit=True
    )
    c = conn.cursor()

    # Check if column exists in MySQL
    c.execute("SHOW COLUMNS FROM attendance LIKE 'subject'")
    col = c.fetchone()

    if not col:
        c.execute("ALTER TABLE attendance ADD COLUMN subject VARCHAR(255)")
        print("🟢 Added 'subject' column to attendance table")

    conn.close()




# --------------------------------------------------------------------
# FACE RECOGNIZER INITIALIZATION
# --------------------------------------------------------------------
def init_face_recognizer():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Try to get OpenCV LBPH recognizer, else fall back to simple one
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        try:
            recognizer = cv2.face_LBPHFaceRecognizer.create()
        except AttributeError:
            try:
                recognizer = cv2.face.LBPHFaceRecognizer.create()
            except AttributeError:
                # Fallback custom recognizer
                class SimpleFaceRecognizer:
                    def __init__(self):
                        self.labels = []
                        self.faces = []

                    def train(self, faces, labels):
                        self.faces = faces
                        self.labels = labels

                    def predict(self, face):
                        if not self.faces:
                            return (-1, 100)
                        min_dist = float('inf')
                        best_label = -1
                        for i, trained_face in enumerate(self.faces):
                            hist1 = cv2.calcHist([face], [0], None, [256], [0, 256])
                            hist2 = cv2.calcHist([trained_face], [0], None, [256], [0, 256])
                            dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                            dist = 1 - dist
                            if dist < min_dist:
                                min_dist = dist
                                best_label = self.labels[i]
                        confidence = min_dist * 100
                        return (best_label, confidence)

                    def save(self, filename):
                        data = {
                            'faces': [f.tolist() for f in self.faces],
                            'labels': self.labels
                        }
                        with open(filename, 'w') as f:
                            json.dump(data, f)

                    def read(self, filename):
                        if os.path.exists(filename):
                            with open(filename, 'r') as f:
                                data = json.load(f)
                            self.faces = [np.array(f) for f in data['faces']]
                            self.labels = data['labels']

                recognizer = SimpleFaceRecognizer()

    # Load cascade classifier
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except Exception:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # If a model already exists (server restarted), load it
    if os.path.exists(MODEL_PATH):
        try:
            if hasattr(recognizer, 'read'):
                recognizer.read(MODEL_PATH)
            elif hasattr(recognizer, 'load'):
                recognizer.load(MODEL_PATH)
            else:
                recognizer.read(MODEL_PATH)
            print("✅ Face model loaded from disk.")
        except Exception as e:
            print("⚠ Could not load existing model:", e)

    return recognizer, face_cascade

recognizer, face_cascade = init_face_recognizer()

# --------------------------------------------------------------------
# FORMS
# --------------------------------------------------------------------
class TeacherLoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class StudentLoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class StudentRegistrationForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    class_name = StringField('Class', validators=[DataRequired()])
    roll_number = StringField('Roll Number', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Register')

class MarksForm(FlaskForm):
    student_id = IntegerField('Student ID', validators=[DataRequired()])
    subject = StringField('Subject', validators=[DataRequired()])
    marks = IntegerField('Marks Obtained', validators=[DataRequired()])
    max_marks = IntegerField('Maximum Marks', validators=[DataRequired()])
    exam_date = StringField('Exam Date (YYYY-MM-DD)', validators=[DataRequired()])
    submit = SubmitField('Upload Marks')

# --------------------------------------------------------------------
# ROUTES: AUTH / DASHBOARDS
# --------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    form = TeacherLoginForm()
    if form.validate_on_submit():
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM teachers WHERE email = %s", (form.email.data,))
        teacher = c.fetchone()
        c.close()
        conn.close()

        if teacher and check_password_hash(teacher[3], form.password.data):
            session['teacher_id'] = teacher[0]
            session['teacher_name'] = teacher[1]
            session['user_type'] = 'teacher'
            flash('Login successful!', 'success')
            return redirect(url_for('teacher_dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('teacher_login.html', form=form)

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    form = StudentLoginForm()
    if form.validate_on_submit():
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM students WHERE email = %s AND approved = 1", (form.email.data,))
        student = c.fetchone()
        c.close()
        conn.close()

        if student and check_password_hash(student[7], form.password.data):
            session['student_id'] = student[0]
            session['student_name'] = student[1]
            session['user_type'] = 'student'
            flash('Login successful!', 'success')
            return redirect(url_for('student_dashboard'))
        else:
            flash('Invalid email, password, or account not approved', 'error')

    return render_template('student_login.html', form=form)

@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    form = StudentRegistrationForm()
    if form.validate_on_submit():
        conn = get_connection()
        c = conn.cursor()
        try:
            hashed_password = generate_password_hash(form.password.data)
            c.execute(
                '''INSERT INTO students (name, email, class, roll_number, password)
                   VALUES (%s, %s, %s, %s, %s)''',
                (form.name.data, form.email.data, form.class_name.data,
                 form.roll_number.data, hashed_password)
            )
            conn.commit()
            student_id = c.lastrowid
            session['temp_student_id'] = student_id
            flash('Registration successful! Please complete face registration.', 'success')
            return redirect(url_for('face_registration'))
        except IntegrityError:
            conn.rollback()
            flash('Email or Roll Number already exists!', 'error')
        finally:
            c.close()
            conn.close()

    return render_template('student_register.html', form=form)

@app.route('/face_registration')
def face_registration():
    if 'temp_student_id' not in session:
        flash('Please register first.', 'error')
        return redirect(url_for('student_register'))
    return render_template('face_registration.html')

# --------------------------------------------------------------------
# FACE CAPTURE & TRAINING
# --------------------------------------------------------------------
@app.route('/capture_face', methods=['POST'])
def capture_face():
    data = request.get_json()
    image_data = data.get('image', '').split(',')[1] if data and 'image' in data else None
    student_id = session.get('temp_student_id')

    if not student_id:
        return jsonify({'success': False, 'message': 'Session expired. Please register again.'})

    if not image_data:
        return jsonify({'success': False, 'message': 'No image data received.'})

    img_data = base64.b64decode(image_data)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image data.'})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected. Please try again.'})

    face_dir = os.path.join(TRAINING_DIR, f'student_{student_id}')
    os.makedirs(face_dir, exist_ok=True)

    face_count = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
    filename = os.path.join(face_dir, f'face_{face_count + 1}.jpg')

    # Save only first detected face in frame
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    cv2.imwrite(filename, face_roi)

    return jsonify({'success': True,
                    'message': f'Face captured successfully! ({face_count + 1}/20)'})

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train on ALL students in training_images/student_<id> folders.
    Called from JS after face registration.
    """
    try:
        faces = []
        labels = []
        label_map = {}
        current_label = 0

        if not os.path.exists(TRAINING_DIR):
            return jsonify({"success": False,
                            "message": "No training images found. Please capture faces first."})

        for folder in os.listdir(TRAINING_DIR):
            folder_path = os.path.join(TRAINING_DIR, folder)
            if not os.path.isdir(folder_path):
                continue

            # expect folder like: student_5
            try:
                student_id = int(folder.replace("student_", ""))
            except ValueError:
                continue

            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
            if len(image_files) == 0:
                continue

            for img_name in image_files:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(current_label)

            label_map[current_label] = student_id
            current_label += 1

        if len(faces) < 10:
            return jsonify({"success": False,
                            "message": "Not enough images to train the model. Capture more faces."})

        recognizer.train(faces, np.array(labels))

        # Save model
        if hasattr(recognizer, 'write'):
            recognizer.write(MODEL_PATH)
        elif hasattr(recognizer, 'save'):
            recognizer.save(MODEL_PATH)
        else:
            recognizer.save(MODEL_PATH)

        # Save label map
        np.save(LABEL_MAP_PATH, label_map)

        # clear temp registration session if present
        session.pop('temp_student_id', None)

        return jsonify({"success": True,
                        "message": "Model trained successfully for all students!"})

    except Exception as e:
        print("TRAIN MODEL ERROR:", e)
        return jsonify({"success": False, "message": "Training failed on server."}), 500

# --------------------------------------------------------------------
# DASHBOARDS
# --------------------------------------------------------------------
@app.route('/teacher/dashboard')
def teacher_dashboard():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM students WHERE approved = 0")
    pending_approvals = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM students WHERE approved = 1")
    total_students = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = %s", (date.today(),))
    today_attendance = c.fetchone()[0]

    # recent 5 attendance entries
    c.execute("""
        SELECT s.name, a.date, a.subject, a.status 
        FROM attendance a 
        JOIN students s ON a.student_id = s.id
        ORDER BY a.timestamp DESC LIMIT 5
    """)
    recent_attendance = c.fetchall()

    c.close()
    conn.close()

    return render_template('teacher_dashboard.html',
                           pending_approvals=pending_approvals,
                           total_students=total_students,
                           today_attendance=today_attendance,
                           recent_attendance=recent_attendance)


@app.route('/student/dashboard')
def student_dashboard():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    conn = get_connection()
    c = conn.cursor()

    c.execute('''SELECT status, COUNT(*) FROM attendance 
                 WHERE student_id = %s GROUP BY status''', (student_id,))
    attendance_data = dict(c.fetchall())

    c.execute('''SELECT subject, marks, max_marks, exam_date FROM marks 
                 WHERE student_id = %s ORDER BY exam_date DESC LIMIT 5''',
              (student_id,))
    recent_marks = c.fetchall()
    c.close()
    conn.close()

    return render_template('student_dashboard.html',
                           attendance_data=attendance_data,
                           recent_marks=recent_marks)

# --------------------------------------------------------------------
# AUTO-RECOGNITION ATTENDANCE
# --------------------------------------------------------------------
@app.route('/teacher/take_attendance')
def take_attendance():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))
    # later we can pass subjects/classes here
    return render_template('take_attendance.html')

from datetime import datetime, date  # make sure this is imported at top

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    # Ensure model exists
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH)):
        return jsonify({'success': False,
                        'message': 'Face model not trained yet. Please train after registrations.'})

    data = request.get_json() or {}
    image_data = data.get('image', '').split(',')[1] if data.get('image') else None
    subject = (data.get('subject') or '').strip() or None
    date_str = data.get('date')

    # Parse date from frontend or default to today
    att_date = date.today()
    if date_str:
        try:
            att_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            pass  # if invalid, still use today

    if not image_data:
        return jsonify({'success': False, 'message': 'No image data received.'})

    # Decode base64 image
    img_data = base64.b64decode(image_data)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image.'})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)  # a bit bigger face region for stability
    )

    if len(faces) == 0:
        return jsonify({'success': True, 'recognized_students': []})

    # label map
    try:
        label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    except Exception as e:
        print("Error loading label map:", e)
        return jsonify({'success': False, 'message': 'Label map missing or corrupted.'})


    conn = pymysql.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB,
    autocommit=True
)

    c = conn.cursor()

    recognized_students = []
    already_in_this_frame = set()  # avoid same student twice in one frame

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        try:
            predicted_label, confidence = recognizer.predict(face_roi)
            student_id = label_map.get(predicted_label)

            if student_id is None:
                continue

            # lower confidence value means better match in LBPH
            if confidence <= ATTENDANCE_CONFIDENCE_THRESHOLD:
                if student_id in already_in_this_frame:
                    continue
                already_in_this_frame.add(student_id)

                c.execute("SELECT name, approved FROM students WHERE id = %s", (student_id,))
                student = c.fetchone()
                if not student or student[1] != 1:
                    continue  # skip unapproved

                # avoid duplicate attendance for same student/date/subject
                if subject:
                    c.execute('''SELECT id FROM attendance 
                                 WHERE student_id = %s AND date = %s AND subject = %s''',
                              (student_id, att_date, subject))
                else:
                    c.execute('''SELECT id FROM attendance 
                                 WHERE student_id = %s AND date = %s AND subject IS NULL''',
                              (student_id, att_date))
                existing = c.fetchone()

                if not existing:
                    c.execute('''INSERT INTO attendance (student_id, date, subject, status)
                                 VALUES (%s, %s, %s, %s)''',
                              (student_id, att_date, subject, 'Present'))
                    conn.commit()

                recognized_students.append({
                    'id': student_id,
                    'name': student[0],
                    'confidence': round(confidence, 2)
                })

        except Exception as e:
            print(f"Recognition error: {e}")
            continue

    conn.close()
    return jsonify({'success': True, 'recognized_students': recognized_students})


# --------------------------------------------------------------------
# MANUAL ATTENDANCE (Present / Absent / Late)
# --------------------------------------------------------------------
@app.route('/teacher/manual_attendance', methods=['GET', 'POST'])
def manual_attendance():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, name, class, roll_number
        FROM students
        WHERE approved = 1
        ORDER BY class, roll_number, name
    """)
    students = c.fetchall()

    if request.method == 'POST':
        date_str = request.form.get('date')
        subject = request.form.get('subject') or None

        if not date_str:
            flash('Please select a date for attendance.', 'error')
        else:
            try:
                attendance_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                attendance_date = date.today()

            # Loop over all students, read status from form
            for s in students:
                sid = s[0]
                status = request.form.get(f'status_{sid}')  # "Present"/"Absent"/"Late" or None
                if not status:
                    continue

                # Check if already exists for this date+subject
                if subject:
                    c.execute(
                        "SELECT id FROM attendance WHERE student_id=%s AND date=%s AND subject=%s",
                        (sid, attendance_date, subject)
                    )
                else:
                    c.execute(
                        "SELECT id FROM attendance WHERE student_id=%s AND date=%s AND subject IS NULL",
                        (sid, attendance_date)
                    )
                row = c.fetchone()

                if row:
                    # Update status
                    c.execute("UPDATE attendance SET status=%s WHERE id=%s", (status, row[0]))
                else:
                    c.execute(
                        "INSERT INTO attendance (student_id, date, subject, status) VALUES (%s,%s,%s,%s)",
                        (sid, attendance_date, subject, status)
                    )

            conn.commit()
            flash('Manual attendance saved successfully!', 'success')
            return redirect(url_for('manual_attendance'))

    c.close()
    conn.close()
    # date default: today – easier for template
    return render_template('manual_attendance.html', students=students, today=date.today())

# --------------------------------------------------------------------
# MARKS / ANALYTICS / APPROVAL
# --------------------------------------------------------------------
@app.route('/teacher/upload_marks', methods=['GET', 'POST'])
def upload_marks():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))
    
    form = MarksForm()

    conn = get_connection()
    c = conn.cursor()

    if form.validate_on_submit():
        # Check student exists & approved
        c.execute("SELECT id FROM students WHERE id = %s AND approved = 1", (form.student_id.data,))
        if not c.fetchone():
            flash('Student ID not found or not approved', 'error')
        else:
            c.execute(
                '''INSERT INTO marks (student_id, subject, marks, max_marks, exam_date)
                   VALUES (%s, %s, %s, %s, %s)''',
                (form.student_id.data, form.subject.data, form.marks.data,
                 form.max_marks.data, form.exam_date.data)
            )
            conn.commit()
            flash('Marks uploaded successfully!', 'success')

    # Fetch most recent 5 marks
    c.execute("""
        SELECT s.name, m.subject, m.marks, m.max_marks, m.exam_date
        FROM marks m
        JOIN students s ON m.student_id = s.id
        ORDER BY m.exam_date DESC LIMIT 5
    """)
    recent_marks = c.fetchall()

    c.close()
    conn.close()
    return render_template('upload_marks.html', form=form, recent_marks=recent_marks)


@app.route('/teacher/analytics')
def analytics():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()

    attendance_df = pd.read_sql('''
        SELECT s.name, s.class, a.date, a.subject, a.status 
        FROM attendance a 
        JOIN students s ON a.student_id = s.id 
        WHERE s.approved = 1
    ''', conn)

    marks_df = pd.read_sql('''
        SELECT s.name, s.class, m.subject, m.marks, m.max_marks, m.exam_date
        FROM marks m 
        JOIN students s ON m.student_id = s.id 
        WHERE s.approved = 1
    ''', conn)

    conn.close()
    charts = generate_analytics_charts(attendance_df, marks_df)
    return render_template('analytics.html', charts=charts)

def generate_analytics_charts(attendance_df, marks_df):
    charts = {}
    try:
        # Attendance trend
        if not attendance_df.empty:
            plt.figure(figsize=(10, 6))
            daily_attendance = attendance_df[attendance_df['status'] == 'Present'] \
                .groupby('date').size()
            if not daily_attendance.empty:
                daily_attendance.plot(kind='line')
                plt.title('Daily Attendance Trend')
                plt.xlabel('Date')
                plt.ylabel('Number of Students')
                plt.xticks(rotation=45)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                charts['attendance_trend'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

        # Attendance by class
        if not attendance_df.empty:
            plt.figure(figsize=(10, 6))
            class_attendance = attendance_df[attendance_df['status'] == 'Present'] \
                .groupby('class').size()
            if not class_attendance.empty:
                class_attendance.plot(kind='bar')
                plt.title('Attendance by Class')
                plt.xlabel('Class')
                plt.ylabel('Number of Present Students')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                charts['class_attendance'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

        # Marks distribution
        if not marks_df.empty:
            plt.figure(figsize=(10, 6))
            marks_df['percentage'] = (marks_df['marks'] / marks_df['max_marks']) * 100
            plt.hist(marks_df['percentage'], bins=20, alpha=0.7)
            plt.title('Marks Distribution')
            plt.xlabel('Percentage')
            plt.ylabel('Number of Students')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts['marks_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
    except Exception as e:
        print(f"Error generating charts: {e}")
    return charts

@app.route('/teacher/approval_queue')
def approval_queue():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    c = conn.cursor()
    c.execute('''SELECT id, name, email, class, roll_number, created_at 
                 FROM students WHERE approved = 0''')
    pending_students = c.fetchall()
    c.close()
    conn.close()

    return render_template('approval_queue.html', pending_students=pending_students)

@app.route('/teacher/approve_student/<int:student_id>')
def approve_student(student_id):
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE students SET approved = 1 WHERE id = %s", (student_id,))
    conn.commit()
    c.close()
    conn.close()

    flash('Student approved successfully!', 'success')
    return redirect(url_for('approval_queue'))

@app.route('/teacher/reject_student/<int:student_id>')
def reject_student(student_id):
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id = %s", (student_id,))
    conn.commit()
    c.close()
    conn.close()

    # Delete training images
    import shutil
    face_dir = os.path.join(TRAINING_DIR, f'student_{student_id}')
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)

    flash('Student rejected and data deleted!', 'success')
    return redirect(url_for('approval_queue'))

# --------------------------------------------------------------------
# EXPORT TO EXCEL
# --------------------------------------------------------------------
@app.route('/export/attendance')
def export_attendance():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    df = pd.read_sql('''
        SELECT s.name, s.class, s.roll_number, a.date, a.subject, a.status, a.timestamp
        FROM attendance a 
        JOIN students s ON a.student_id = s.id 
        WHERE s.approved = 1
        ORDER BY a.date DESC, s.class, s.name
    ''', conn)
    conn.close()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Attendance', index=False)
    output.seek(0)
    return send_file(output, download_name='attendance_records.xlsx', as_attachment=True)

@app.route('/export/marks')
def export_marks():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    conn = get_connection()
    df = pd.read_sql('''
        SELECT s.name, s.class, s.roll_number, m.subject, m.marks, m.max_marks, m.exam_date
        FROM marks m 
        JOIN students s ON m.student_id = s.id 
        WHERE s.approved = 1
        ORDER BY m.exam_date DESC, s.class, s.name
    ''', conn)
    conn.close()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Marks', index=False)
    output.seek(0)
    return send_file(output, download_name='marks_records.xlsx', as_attachment=True)


@app.route("/api/get_student/<int:student_id>")
def api_get_student(student_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name, class FROM students WHERE id = %s AND approved = 1", (student_id,))
    student = c.fetchone()
    c.close()
    conn.close()

    if student:
        return jsonify({"success": True, "name": student[0], "class": student[1]})
    return jsonify({"success": False})

# --------------------------------------------------------------------
# LOGOUT & MAIN
# --------------------------------------------------------------------
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    init_db()
    migrate_db()  # <-- ADD THIS

    # Create default teacher account if not exists
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM teachers WHERE email = 'teacher@school.com'")
    if not c.fetchone():
        hashed_password = generate_password_hash('teacher123')
        c.execute(
            "INSERT INTO teachers (name, email, password) VALUES (%s, %s, %s)",
            ('Admin Teacher', 'teacher@school.com', hashed_password)
        )
        conn.commit()
    c.close()
    conn.close()

    print("\n🚀 Smart Attendance System starting...")
    print("📧 Teacher Login: teacher@school.com / teacher123")
    print("🌐 Access: http://localhost:5000\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
