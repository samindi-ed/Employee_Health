from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import os
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure MySQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:976752678@localhost/employee_db'  # Update with correct DB name and credentials
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the CBC model
cbc_model = joblib.load('cbc_model.pkl')

# Load the BMP model
bmp_model = joblib.load('bmp_model.pkl')
# Load the other models
lipid_panel_model = joblib.load('lipid_panel_model.pkl')
thyroid_panel_model = joblib.load('thyroid_panel_model.pkl')
cardiac_biomarkers_model = joblib.load('cardiac_biomarkers_model.pkl')
ecg_values_model = joblib.load('ecg_values_model.pkl')

# Define the general features
general_features = ['age', 'height', 'weight', 'BMI']
lipid_features = ['HDL', 'LDL']
thyroid_features = ['T3', 'T4', 'TSH']
cardiac_features = ['hs_cTn', 'BNP', 'NT_proBNP', 'CK', 'CK_MB']
ecg_features = ['RR_interval', 'P_wave', 'PR_interval', 'PR_segment', 'QRS_complex', 'ST_segment', 'T_wave', 'QT_interval']

# Define the features for the CBC section
cbc_features = ['red_blood_cells', 'white_blood_cells', 'platelets', 'hemoglobin', 'hematocrit']
# Define the features for the BMP section
bmp_features = ['BUN', 'creatinine', 'glucose', 'CO2', 'calcium', 'sodium', 'potassium', 'chloride']

# Define the Employee model
class Employee(db.Model):
    __tablename__ = 'employee'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)

# Define the EmployeeHealth model
class EmployeeHealth(db.Model):
    __tablename__ = 'employee_health'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    age = db.Column(db.Float)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    cbc_rbc = db.Column(db.Float)
    cbc_wbc = db.Column(db.Float)
    cbc_platelets = db.Column(db.Float)
    cbc_hemoglobin = db.Column(db.Float)
    cbc_hematocrit = db.Column(db.Float)
    bmp_bun = db.Column(db.Float)
    bmp_creatinine = db.Column(db.Float)
    bmp_glucose = db.Column(db.Float)
    bmp_co2 = db.Column(db.Float)
    bmp_calcium = db.Column(db.Float)
    bmp_sodium = db.Column(db.Float)
    bmp_potassium = db.Column(db.Float)
    bmp_chloride = db.Column(db.Float)
    lipid_hdl = db.Column(db.Float)
    lipid_ldl = db.Column(db.Float)
    thyroid_t3 = db.Column(db.Float)
    thyroid_t4 = db.Column(db.Float)
    thyroid_tsh = db.Column(db.Float)
    cardiac_hs_ctn = db.Column(db.Float)
    cardiac_bnp = db.Column(db.Float)
    cardiac_nt_probnp = db.Column(db.Float)
    cardiac_ck = db.Column(db.Float)
    cardiac_ck_mb = db.Column(db.Float)
    ecg_rr_interval = db.Column(db.Float)
    ecg_p_wave = db.Column(db.Float)
    ecg_pr_interval = db.Column(db.Float)
    ecg_pr_segment = db.Column(db.Float)
    ecg_qrs_complex = db.Column(db.Float)
    ecg_st_segment = db.Column(db.Float)
    ecg_t_wave = db.Column(db.Float)
    ecg_qt_interval = db.Column(db.Float)

    employee = db.relationship('Employee', backref=db.backref('health_records', lazy=True))


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index', methods=['POST'])
def index():
    employee_id = request.form['employee_id']
    session['employee_id'] = employee_id
    mandatory_params = general_features
    sections = {
        "Complete Blood Count": cbc_features,
        "Basic Metabolic Panel": bmp_features,
        "Lipid Panel": lipid_features,
        "Thyroid Panel": thyroid_features,
        "Cardiac Biomarkers": cardiac_features,
        "ECG Values": ecg_features
    }
    return render_template('index.html', employee_id=employee_id, mandatory_params=mandatory_params, sections=sections)

@app.route('/result', methods=['POST'])
def result():
    employee_id = session.get('employee_id')
    all_params = request.form.to_dict()
    section = request.form.get('section')

    def get_float_value(param):
        value = all_params.get(param)
        try:
            return float(value) if value else None
        except ValueError:
            return None

    general_data = {param: get_float_value(param) for param in general_features}
    section_results = {}
    recommendations = []

    def process_section(section_name, section_features, model):
        section_data = {param: get_float_value(param) for param in section_features}
        if any(section_data.values()):
            data = {**general_data, **section_data}
            df = pd.DataFrame(data, index=[0])
            missing_cols = set(general_features + section_features) - set(df.columns)
            for col in missing_cols:
                df[col] = 0.0
            df = df[general_features + section_features]
            prediction = model.predict(df)[0]
            if prediction == 1:
                out_of_range_biomarkers = []
                for param, (low, high) in zip(section_features, section_ranges[section_name]):
                    if section_data[param] < low or section_data[param] > high:
                        out_of_range_biomarkers.append(param)
                        recommendations.append(f"Possible issues related to {param}.")
                if out_of_range_biomarkers:
                    recommendations.append(f"Out-of-range biomarkers detected: {', '.join(out_of_range_biomarkers)}.")
                    recommendations.append("It is recommended to consult a doctor for further evaluation.")
                    section_results[section_name] = 'bad'
            else:
                section_results[section_name] = 'good'

    section_ranges = {
        "Complete Blood Count": [(4.5, 6.1), (4.0, 10.8), (150, 400), (13.0, 17.0), (40, 52)],
        "Basic Metabolic Panel": [(6, 20), (0.6, 1.3), (70, 100), (23, 29), (8.5, 10.2), (135, 145), (3.7, 5.2), (96, 106)],
        "Lipid Panel": [(60, float('inf')), (float('-inf'), 100)],
        "Thyroid Panel": [(80, 180), (0.8, 1.8), (0.5, 4)],
        "Cardiac Biomarkers": [(float('-inf'), 1), (float('-inf'), 100), (float('-inf'), 300), (30, 200), (0, 12)],
        "ECG Values": [(0.6, 1.2), (80, 80), (120, 200), (50, 120), (80, 100), (80, 120), (160, 160), (float('-inf'), 420)]
    }

    # Process only the specific section or all sections if 'Overall' is chosen
    if section == "Complete Blood Count":
        process_section("Complete Blood Count", cbc_features, cbc_model)
    elif section == "Basic Metabolic Panel":
        process_section("Basic Metabolic Panel", bmp_features, bmp_model)
    elif section == "Lipid Panel":
        process_section("Lipid Panel", lipid_features, lipid_panel_model)
    elif section == "Thyroid Panel":
        process_section("Thyroid Panel", thyroid_features, thyroid_panel_model)
    elif section == "Cardiac Biomarkers":
        process_section("Cardiac Biomarkers", cardiac_features, cardiac_biomarkers_model)
    elif section == "ECG Values":
        process_section("ECG Values", ecg_features, ecg_values_model)
    elif section == "Overall":
        for section_name, section_features, model in [
            ("Complete Blood Count", cbc_features, cbc_model),
            ("Basic Metabolic Panel", bmp_features, bmp_model),
            ("Lipid Panel", lipid_features, lipid_panel_model),
            ("Thyroid Panel", thyroid_features, thyroid_panel_model),
            ("Cardiac Biomarkers", cardiac_features, cardiac_biomarkers_model),
            ("ECG Values", ecg_features, ecg_values_model)
        ]:
            process_section(section_name, section_features, model)

    health_status = 'Good' if all(status == 'good' for status in section_results.values()) else 'Bad'
    prediction_text = 'Overall health status is good.' if health_status == 'Good' else 'There are some health issues.'

    # Check if the employee exists
    employee = Employee.query.get(employee_id)
    if not employee:
        # If employee does not exist, create a new employee
        name = request.form.get('employee_name', 'Unknown')  # Default name if not provided
        position = request.form.get('employee_position', 'Unknown')  # Default position if not provided
        new_employee = Employee(id=employee_id, name=name, position=position)
        db.session.add(new_employee)
        db.session.commit()

    # Save the health data
    new_employee_health = EmployeeHealth(
        employee_id=employee_id,
        age=get_float_value('age'),
        height=get_float_value('height'),
        weight=get_float_value('weight'),
        bmi=get_float_value('BMI'),
        cbc_rbc=get_float_value('red_blood_cells'),
        cbc_wbc=get_float_value('white_blood_cells'),
        cbc_platelets=get_float_value('platelets'),
        cbc_hemoglobin=get_float_value('hemoglobin'),
        cbc_hematocrit=get_float_value('hematocrit'),
        bmp_bun=get_float_value('BUN'),
        bmp_creatinine=get_float_value('creatinine'),
        bmp_glucose=get_float_value('glucose'),
        bmp_co2=get_float_value('CO2'),
        bmp_calcium=get_float_value('calcium'),
        bmp_sodium=get_float_value('sodium'),
        bmp_potassium=get_float_value('potassium'),
        bmp_chloride=get_float_value('chloride'),
        lipid_hdl=get_float_value('HDL'),
        lipid_ldl=get_float_value('LDL'),
        thyroid_t3=get_float_value('T3'),
        thyroid_t4=get_float_value('T4'),
        thyroid_tsh=get_float_value('TSH'),
        cardiac_hs_ctn=get_float_value('hs_cTn'),
        cardiac_bnp=get_float_value('BNP'),
        cardiac_nt_probnp=get_float_value('NT_proBNP'),
        cardiac_ck=get_float_value('CK'),
        cardiac_ck_mb=get_float_value('CK_MB'),
        ecg_rr_interval=get_float_value('RR_interval'),
        ecg_p_wave=get_float_value('P_wave'),
        ecg_pr_interval=get_float_value('PR_interval'),
        ecg_pr_segment=get_float_value('PR_segment'),
        ecg_qrs_complex=get_float_value('QRS_complex'),
        ecg_st_segment=get_float_value('ST_segment'),
        ecg_t_wave=get_float_value('T_wave'),
        ecg_qt_interval=get_float_value('QT_interval')
    )

    try:
        db.session.add(new_employee_health)
        db.session.commit()  # Commit the transaction to save the data
    except Exception as e:
        db.session.rollback()
        print(f"Error occurred while saving data: {e}")

    return render_template('result.html', prediction_text=prediction_text, prediction_class=health_status.lower(), recommendations=recommendations, section_results=section_results, employee_id=employee_id)

@app.route('/signout', methods=['POST'])
def signout():
    session.pop('employee_id', None)
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("MySQL Database initialized successfully!")
    app.run(debug=True, host='0.0.0.0', port=8080)
