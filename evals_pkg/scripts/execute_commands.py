import subprocess

with open('commands.txt', 'r') as file:
    commands = file.readlines()

for command in commands:
    command = command.strip()  # Remove any trailing newline characters
    if command:
        process = subprocess.Popen(command, shell=True)
        process.communicate()  # Wait for the command to complete