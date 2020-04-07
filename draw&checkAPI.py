# from modelExploit import predict
import pygame

pygame.init()
window = pygame.display.set_mode((500, 500))
matrix = 289 * [289 * [0]]
background= pygame.image.load("background.png").convert_alpha()
button = pygame.image.load("button.png").convert_alpha()
buttonActive = pygame.image.load("buttonActive.png").convert_alpha()
result = 'NaN'
font = pygame.font.SysFont("Sans-serif", 25)
window.blit(background, (0, 0))

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
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouseX, mouseY = event.pos
            mouseClickedLeft = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            mouseX, mouseY = event.pos
            mouseClickedRight = True
        if mouseX in range(90, 379) and mouseY in range(48, 337) and mouseClickedLeft:
            for i in range(mouseX - 4, mouseX + 4):
                for o in range(mouseY - 4, mouseY + 4):
                    if i - 90 in range(0, 288) and o - 48 in range(0, 288):
                        matrix[i - 90][o - 48] = 1;
                        pygame.draw.rect(window, (255, 255, 255), (90 + i, o + 48, 1, 1))
        if mouseX in range(145, 344) and mouseY in range(360, 405):
            window.blit(font.render("Result: " + str(result), 1, (0, 0, 0)), (200, 440))
            window.blit(buttonActive, (145, 360))
            if mouseClickedLeft:
                ilt = 0
        else:
            window.blit(font.render("Result: " + str(result), 1, (0, 0, 0)), (200, 440))
            window.blit(button, (146, 361))
        pygame.display.update()


# predict("digit.png")