import re
import json
import secrets


def check_usr_pass(username: str, password: str) -> bool:
    """
    Authenticates the username and password.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_user_data = json.load(auth_json)

    for registered_user in authorized_user_data:
        if registered_user['username'] == username:
            try:
                passwd_verification_bool = registered_user['password'] == password
                if passwd_verification_bool == True:
                    return True
            except:
                pass
    return False


def check_valid_name(name_sign_up: str) -> bool:
    """
    Checks if the user entered a valid name while creating the account.
    """
    name_regex = (r'^[A-Za-z\u4e00-\u9fff]*')

    if re.search(name_regex, name_sign_up):
        return True
    return False


def non_empty_str_check(username_sign_up: str) -> bool:
    """
    Checks for non-empty strings.
    """
    empty_count = 0
    for i in username_sign_up:
        if i == ' ':
            empty_count = empty_count + 1
            if empty_count == len(username_sign_up):
                return False

    if not username_sign_up:
        return False
    return True


def check_unique_usr(username_sign_up: str):
    """
    Checks if the username already exists (since username needs to be unique),
    also checks for non - empty username.
    """
    authorized_user_data_master = list()
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            authorized_user_data_master.append(user['username'])

    if username_sign_up in authorized_user_data_master:
        return False
    
    non_empty_check = non_empty_str_check(username_sign_up)

    if non_empty_check == False:
        return None
    return True


def register_new_usr(name_sign_up: str, username_sign_up: str, password_sign_up: str) -> None:
    """
    Saves the information of the new user in the _secret_auth.json file.
    """
    new_usr_data = {'username': username_sign_up, 'name': name_sign_up, 'password': password_sign_up}

    with open("_secret_auth_.json", "r") as auth_json:
        authorized_user_data = json.load(auth_json)

    with open("_secret_auth_.json", "w") as auth_json_write:
        authorized_user_data.append(new_usr_data)
        json.dump(authorized_user_data, auth_json_write)


def check_username_exists(user_name: str) -> bool:
    """
    Checks if the username exists in the _secret_auth.json file.
    """
    authorized_user_data_master = list()
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            authorized_user_data_master.append(user['username'])
        
    if user_name in authorized_user_data_master:
        return True
    return False


def generate_random_passwd() -> str:
    """
    Generates a random password.
    """
    password_length = 10
    return secrets.token_urlsafe(password_length)


def change_passwd(username: str, random_password: str) -> None:
    """
    Replaces the old password with the newly generated password.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

    with open("_secret_auth_.json", "w") as auth_json_:
        for user in authorized_users_data:
            if user['username'] == username:
                user['password'] = random_password
        json.dump(authorized_users_data, auth_json_)
    

def check_current_passwd(reset_passwd: str, current_passwd: str) -> bool:
    """
    Authenticates the password entered against the username when 
    resetting the password.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            if user['email'] == reset_passwd:
                try:
                    if user['password'] == current_passwd:
                        return True
                except:
                    pass
    return False
