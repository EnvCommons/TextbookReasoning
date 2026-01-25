from openreward.environments import Server

from textbookreasoning import TextbookReasoning

if __name__ == "__main__":
    server = Server([TextbookReasoning])
    server.run()
