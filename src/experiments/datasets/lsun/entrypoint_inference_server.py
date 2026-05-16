from llavaguard_on_sglang.sglang_gpt_router import LlavaGuardServer

def main():
    # TODO: Let user pass port, dp_size and model name as commandline args
    try:
        server = LlavaGuardServer()
        server.setUpClass(
            model="AIML-TUDA/LlavaGuard-v1.2-7B-OV",
            dp_size=4,
            port=10001
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt. Shutting down server.")
        server.tearDownClass()

if __name__ == '__main__':
    main()
