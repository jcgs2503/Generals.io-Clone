import pygame
pygame.init()

display_width = 1000
display_height = 800

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Generals.io')

black = (0, 0, 0)
white = (255, 255, 255)

gameExit = False
clock = pygame.time.Clock()

towerImg = pygame.image.load('./images/tower.png')
towerImg = pygame.transform.smoothscale(towerImg, (25, 25))


def tower(x, y):
    gameDisplay.blit(towerImg, (x, y))


def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf', 115)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2), (display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()
    pygame.time.wait(2000)
    game_loop()


def crash():
    message_display('You Crashed')


def game_loop():

    x = (display_width * 0.5)
    y = (display_height * 0.5)
    x_change = 0
    y_change = 0

    while not gameExit:
        gameDisplay.fill(white)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -5
                elif event.key == pygame.K_RIGHT:
                    x_change = 5
                elif event.key == pygame.K_UP:
                    y_change = -5
                elif event.key == pygame.K_DOWN:
                    y_change = 5
            if event.type == pygame.KEYUP:
                x_change = 0
                y_change = 0

        x += x_change
        y += y_change
        tower(x, y)

        if x > display_width - 25 or x < 0:
            crash()

        pygame.display.update()
        clock.tick(60)


game_loop()
pygame.quit()
quit()
