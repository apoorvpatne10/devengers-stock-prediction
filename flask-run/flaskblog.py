from flask import Flask, render_template, url_for, flash, redirect
from forms import RegisterForm
from main_mod import do_stuff
import secrets
secrets.token_hex(16)

app = Flask(__name__)
app.config["SECRET_KEY"] = '531d9739ed8dab2ebb5d1c38f71c1446'

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route('/check_stock', methods=['GET', 'POST'])
def check_stock():
    form = RegisterForm()
    if form.validate_on_submit():
        flash(f"Testing now for {form.phone_no.data}...", "success")
        # print(form.phone_no.data)
        # print(form.threshold.data)
        do_stuff(int(form.threshold.data), form.phone_no.data)
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


# if __name__ == '__main__':
#     app.run(debug=True)
