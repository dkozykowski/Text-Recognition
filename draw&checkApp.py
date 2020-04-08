# Testing App
# this application is only a testing future which allows me to check if the model works correctly

from modelExploit import predict
import pygame
from PIL import Image
import numpy as np

pygame.init()
window = pygame.display.set_mode((500, 500))
matrix = [[0 for x in range(291)] for y in range(291)]
backgroundFilled= pygame.image.load("images/backgroundFilled.png").convert_alpha()
backgroundEmpty= pygame.image.load("images/backgroundEmpty.png").convert_alpha()
buttonCheck = pygame.image.load("images/buttonCheck.png").convert_alpha()
buttonCheckActive = pygame.image.load("images/buttonCheckActive.png").convert_alpha()
buttonReset = pygame.image.load("images/buttonReset.png").convert_alpha()
buttonResetActive = pygame.image.load("images/buttonResetActive.png").convert_alpha()
result = 'NaN'
font = pygame.font.SysFont("Sans-serif", 25)
window.blit(backgroundFilled, (0, 0))
buttonCheckStatus = buttonResetStatus = 0
thickness = 10


def update():
    window.blit(backgroundEmpty, (0, 0))
    window.blit(font.render("Result: " + str(result), 1, (0, 0, 0)), (200, 440))

    # check button
    if buttonCheckStatus == 1:      # button is active
        window.blit(buttonCheckActive, (45, 360))
    else:      # button is inactive
        window.blit(buttonCheck, (46, 361))

    # Reset button
    if buttonResetStatus == 1:      # button is active
        window.blit(buttonResetActive, (245, 360))
    else:      # button is inactive
        window.blit(buttonReset, (246, 361))


while True:
    for event in pygame.event.get():
        mouseClickedRight = mouseClickedLeft = False
        mouseX = mouseY = 0
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)
        if event.type == pygame.MOUSEMOTION:
            # coordinates of mouse cursor
            mouseX, mouseY = event.pos
        if pygame.mouse.get_pressed()[0] and mouseX in range(107, 398) and mouseY in range(47, 338):
            for i in range(mouseX - thickness, mouseX + thickness):
                for o in range(mouseY - thickness, mouseY + thickness):
                    if i - 107 in range(0, 291) and o - 47 in range(0, 291) and i + o - mouseX - mouseY < thickness:
                        matrix[i - 107][o - 47] = 1
                        pygame.draw.rect(window, (255, 255, 255), (i, o, 1, 1))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouseX, mouseY = event.pos
            mouseClickedLeft = True

        # Check button
        if mouseX in range(45, 244) and mouseY in range(360, 405):
            buttonCheckStatus = 1     # button is active
            if mouseClickedLeft:
                img = Image.new('P', (291, 291))
                for i in range(291):
                    for o in range(291):
                        img.putpixel((i, o), matrix[i][o] * 255)
                img = img.resize((28, 28))
                picture = np.array(img, dtype=np.uint8)
                img = Image.fromarray(picture)
                img.save("images/digit.png")
                result = predict("images/digit.png")
        else:
            buttonCheckStatus = 0     # button is inactive

        # Reset button
        if mouseX in range(245, 444) and mouseY in range(360, 405):
            buttonResetStatus = 1
            if mouseClickedLeft:
                for i in range(291):
                    for o in range(291):
                        matrix[i][o] = 0
                window.blit(backgroundFilled, (0, 0))
        else:
            buttonResetStatus = 0
        update()
        pygame.display.update()
