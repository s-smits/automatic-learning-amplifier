#!/bin/bash

# Define the alias command
alias_command="alias python='/opt/homebrew/bin/python3.10'"

# Function to add alias to a file
add_alias() {
    local file=$1
    if [ -f "$file" ]; then
        # Check if the alias already exists
        if grep -Fxq "$alias_command" "$file"; then
            echo "Alias already exists in $file"
        else
            echo "Adding alias to $file"
            echo "$alias_command" >> "$file"
        fi
    else
        echo "$file not found."
    fi
}

# Paths to the potential config files
zshrc="$HOME/.zshrc"
bash_profile="$HOME/.bash_profile"
bashrc="$HOME/.bashrc"

# Add alias to .zshrc
add_alias "$zshrc"

# Add alias to .bash_profile or .bashrc
if [ -f "$bash_profile" ]; then
    add_alias "$bash_profile"
elif [ -f "$bashrc" ]; then
    add_alias "$bashrc"
else
    echo "No bash config file found."
fi

echo "Setup complete. Please restart your terminal or source the config files."