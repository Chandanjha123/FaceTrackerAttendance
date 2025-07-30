from flask import Flask
from flask import render_template, request, jsonify
import subprocess
#  What each one does:

# render_template: Lets you load HTML files (like index.html) from the templates/ folder.

# request: Lets you access data sent from the frontend (like a person's name).

# jsonify: Converts Python dictionaries to JSON to send back to the frontend.

# subprocess: Allows you to run your existing Python scripts like add_Faces.py and test.py as if you were running them from terminal.

app=Flask(__name__) #Creates your Flask app using the current file’s name. This app variable will be used to define all routes.



# @app.route('/'): When someone goes to the main page (/), this function runs.

# def home():: This is the function that runs for the homepage.

# return render_template('index.html'): Loads your index.html file from the templates folder.
@app.route('/')
def home():
    return render_template('index.html')



# @app.route('/add_face', methods=['POST']): This function runs when the frontend sends a POST request to /add_face.

# name = request.form['name']: Gets the person’s name from the frontend form.

# subprocess.run(...): Runs your existing add_Faces.py script and passes the name to it.

# return jsonify(...): Sends a success message back to the frontend.
app.route('/add_face',method=['POST'])
def add_face():
    name=request.ofrm['name']
    subprocess.run(['python','add_faces.py',name])
    return jsonify({'message':'face added'})


# @app.route('/take_attendance', methods=['GET']): This function runs when the frontend calls /take_attendance.

# subprocess.check_output(...): Runs test.py and captures the printed output (which should be a name).

# output.decode().strip(): Converts byte data to a string (removes b'' and newlines).

# jsonify(...): Sends the name and success message back to frontend.
@app.route('/take_attendance', methods=['GET'])
def take_attendance():
    output = subprocess.check_output(['python', 'test.py'])
    name = output.decode().strip()
    return jsonify({'message': 'Attendance Marked!', 'name': name})


if __name__ == '__main__':
    app.run(debug=True)
