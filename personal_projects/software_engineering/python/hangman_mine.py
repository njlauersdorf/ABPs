#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
import pandas
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

start = print("Welcome to hangman!\n")
time.sleep(1)
name = input("Enter your name: ")
print("Hello, " + name + "! Good luck!\n")
time.sleep(2)
print("Time to begin!\n")


#Initiates the parameters for the game
def main():

    global guess_count                  #Number of guesses made (int)
    global display                      #hidden word (str) with letters guessed showing to show player
    global word                         #hidden word (str) with letters guessed removed
    global letters_guessed              #List of letters (str) guessed
    global play_game                    #Request to play game (y/n)
    global word_length                  #Number of letters in word
    global alphabet                     #List of letters (str) that can be guessed
    global word_permanent               #Full word to be guessed (str)

    #Trys to connect and get list of words/abbreviations to randomly choose from
    try:
        url ="https://www.mit.edu/~ecprice/wordlist.10000"
        http = urllib3.PoolManager()
        response=http.request('GET', url)
        txt = response.data

        word_options = txt.splitlines()
        word = random.choice(word_options).decode("utf-8")

    #If error, use a short, default list
    except urllib3.exceptions.HTTPError as err:
        print('Request to connect to online dictionary failed. Using local word list for Hangman words.')
        word_options = ["python", "is", "fun", "to", "learn"]
        word = random.choice(word_options)

    #Assign initial global variable values
    word_permanent = word
    word_length = len(word)
    guess_count = 0
    display = '_' * word_length
    letters_guessed = []
    play_game = ""
    alphabet = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"

    #Output hidden word with letters as underscores
    time.sleep(2)
    print("Letters remaining to guess:")
    print(alphabet + "\n")

# A loop to re-execute the game when the first round ends:
def play_again():
    global play_game
    play_game = input("Do you want to play again (y/n)?  \n")

    #Loop to modify player's response to desired input (y/n)
    while play_game not in ["y", "n","Y","N"]:
        time.sleep(1)
        print("Please enter either 'y' for 'yes' or 'n' for 'no'")
        play_game = input("Do You want to play again? (y/n)\n")

    #Either play another round (play_game='y') or end the loop (play_game='n')
    if play_game == "y":
        time.sleep(1)
        main()
        hangman()
    elif play_game == "n":
        time.sleep(1)
        print("Peace out, nerd! ( ͡° ͜ʖ ͡°)")
        exit()


# Executes the hangman game for a single round
def hangman():

    #Read in global variables set in main()
    global guess_count
    global display
    global word
    global letters_guessed
    global word_length
    global play_game
    global alphabet
    global word_permanent

    guess_limit = 6
    time.sleep(1)

    #Return progress and number of guesses remaining after each guess
    if guess_count == guess_limit:
        guess = input("Hangman word: " + display + ". You have " + str(guess_limit - guess_count) + " guesses. Enter your letter to guess: \n")
    else:
        guess = input("Hangman word: " + display + ". You have " + str(guess_limit - guess_count) + " guesses remaining. Enter your letter to guess: \n")

    guess = guess.strip()
    time.sleep(1)

    #Make sure guess is appropriate length (1), data type (str), and is contained in the alphabet list variable.
    if isinstance(guess, str):
        if len(guess.strip())!=1 or guess.strip().isalpha()==0:
            print("Invalid Input, Try a letter \n")
            hangman()
        elif guess.lower() in letters_guessed:
            print("You've already guessed that letter! Try another choice from those remaining:")
            print(alphabet + "\n")
        elif guess.lower() in word:
            letters_guessed.append(guess.lower())
            index_remove = alphabet.find(guess.upper())
            alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
            index = word.find(guess.lower())

            while index != -1:
                word = word[:index] + "_" + word[index + 1:]
                display = display[:index] + guess.lower() + display[index + 1:]
                index = word.find(guess.lower())

            time.sleep(1)
            print("Correct guess! \n")
            time.sleep(1)

            print("Letters remaining to guess:")
            print(alphabet + "\n")



        else:
            guess_count += 1

            if guess_count == 1:
                letters_guessed.append(guess.lower())
                index_remove = alphabet.find(guess.upper())
                alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
                time.sleep(1)
                print("   _____ \n"
                      "  |     | \n"
                      "  |     |\n"
                      "  |     O \n"
                      "  |       \n"
                      "  |      \n"
                      "  |      \n"
                      "__|__\n")
                print("Wrong guess. \n")
                time.sleep(1)
                print("Letters remaining to guess:")
                print(alphabet + "\n")

            elif guess_count == 2:
                letters_guessed.append(guess.lower())
                index_remove = alphabet.find(guess.upper())
                alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
                time.sleep(1)
                print("   _____ \n"
                      "  |     | \n"
                      "  |     |\n"
                      "  |     O \n"
                      "  |     | \n"
                      "  |        \n"
                      "  |        \n"
                      "__|__\n")
                print("Wrong guess.\n")
                time.sleep(1)
                print("Letters remaining to guess:")
                print(alphabet + "\n")

            elif guess_count == 3:
               letters_guessed.append(guess.lower())
               index_remove = alphabet.find(guess.upper())
               alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
               time.sleep(1)
               print("   _____ \n"
                     "  |     | \n"
                     "  |     |\n"
                     "  |     O \n"
                     "  |    /| \n"
                     "  |        \n"
                     "  |        \n"
                     "__|__\n")
               print("Wrong guess.\n")
               time.sleep(1)
               print("Letters remaining to guess:")
               print(alphabet + "\n")


            elif guess_count == 4:
                letters_guessed.append(guess.lower())
                index_remove = alphabet.find(guess.upper())
                alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
                time.sleep(1)
                print("   _____ \n"
                      "  |     | \n"
                      "  |     |\n"
                      "  |     O \n"
                      "  |    /|\ \n"
                      "  |        \n"
                      "  |        \n"
                      "__|__\n")
                print("Wrong guess.\n")
                time.sleep(1)
                print("Letters remaining to guess:")
                print(alphabet + "\n")

            elif guess_count == 5:
                letters_guessed.append(guess.lower())
                index_remove = alphabet.find(guess.upper())
                alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
                time.sleep(1)
                print("   _____ \n"
                      "  |     | \n"
                      "  |     |\n"
                      "  |     O \n"
                      "  |    /|\ \n"
                      "  |    /   \n"
                      "  |        \n"
                      "__|__\n")
                print("Wrong guess.\n")
                time.sleep(1)
                print("Letters remaining to guess:")
                print(alphabet + "\n")

            elif guess_count == 6:
                letters_guessed.append(guess.lower())
                index_remove = alphabet.find(guess.upper())
                alphabet = alphabet[:index_remove] + " " + alphabet[index_remove +1:]
                time.sleep(1)
                print("   _____ \n"
                      "  |     | \n"
                      "  |     |\n"
                      "  |     O \n"
                      "  |    /|\ \n"
                      "  |    / \ \n"
                      "  |        \n"
                      "__|__\n")
                print("Wrong guess. You are hanged!!!\n")
                time.sleep(1)
                print("The word was:", word_permanent + "\n")
                play_again()

        if word == '_' * word_length:
            time.sleep(1)
            print("Congrats! You have guessed the word correctly!")
            play_again()

        elif guess_count != guess_limit:
            hangman()
    else:
        print("Invalid Input, Try a letter\n")
        hangman()







main()


hangman()
