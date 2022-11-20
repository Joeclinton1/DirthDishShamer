

class EmailShamer():
    def __init__(self, name_to_email):
        self.name_to_email = name_to_email

    def shame(self, name):
        self.send_email(self.name_to_email[name])

    def send_email(self, email_address):
        import smtplib

        gmail_user = 'dishshamebot@gmail.com'
        gmail_password = 'GOOshamebot1$'

        sent_from = gmail_user
        to = [email_address]
        subject = 'Lorem ipsum dolor sit amet'
        body = 'consectetur adipiscing elit'

        email_text = """\
        From: %s
        To: %s
        Subject: %s

        %s
        """ % (sent_from, ", ".join(to), subject, body)

        try:
            smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            smtp_server.ehlo()
            smtp_server.login(gmail_user, gmail_password)
            smtp_server.sendmail(sent_from, to, email_text)
            smtp_server.close()
            print("Email sent successfully!")
        except Exception as ex:
            print("Something went wrong….", ex)

if __name__ == "__main__":
    email_shamer = EmailShamer({
        "joe": "joeclinton1@btinternet.com"
    })

    email_shamer.shame("joe")

