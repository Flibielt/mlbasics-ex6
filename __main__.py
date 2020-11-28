from ex6.src import ex6, ex6_spam


def main():
    """
    Machine Learning Class - Exercise 6 | Support Vector Machines
    """
    #  Part 1
    #  Support Vector Machines
    ex6()

    if input('Press ENTER to start the next part. (press [q] to exit here)\n') == 'q':
        print('Exit')
        exit(0)

    #  Part 2
    #  Spam Classification with SVMs
    ex6_spam()


if __name__ == '__main__':
    main()
