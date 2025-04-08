import aiohttp
import asyncio
import time

# The text to be converted to speech
text = """Oh man... okay, so—there was this one time I, uh, totally bombed a group presentation 'cause I thought it was next week.. <groan>
I showed up in, like, pajama pants... no slides... and just stood there smiling like a deer in headlights..
My group? Yeah—they never let me live it down.. <chuckle>"""

# List of different voices to use
voices = ["zoe", "zac", "jess", "leo", "mia"]

async def send_tts_request(session, voice, text, request_id):
    """Send a request to the TTS API with the given voice and text"""
    start_time = time.monotonic()
    print(f"[{request_id}] Starting request with voice: {voice}")
    
    params = {
        "prompt": text,
        "voice": voice
    }
    
    try:
        # Send the request to the TTS API
        async with session.get("http://localhost:8000/tts", params=params) as response:
            if response.status == 200:
                print(f"[{request_id}] Completed request with voice: {voice}")
                return voice, None
            else:
                print(f"[{request_id}] Error: {response.status} - {await response.text()}")
                return voice, None
    except Exception as e:
        print(f"[{request_id}] Exception for voice {voice}: {str(e)}")
        return voice, None

async def main():
    """Send 10 concurrent requests to the TTS API with different voices"""
    start_all = time.monotonic()
    
    print(f"Sending 10 concurrent requests to TTS API with different voices")
    print(f"Text: {text[:50]}...")
    
    async with aiohttp.ClientSession() as session:
        # Create a list of tasks for concurrent execution
        tasks = []
        for i in range(10):
            task = send_tts_request(session, voices[i % len(voices)], text, i+1)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
    
    # Print summary
    total_time = time.monotonic() - start_all
    print("\nSummary:")
    for voice, elapsed in results:
        if elapsed:
            print(f"Voice {voice}: {elapsed:.2f} seconds")
        else:
            print(f"Voice {voice}: Failed")
    
    print(f"\nTotal time for all requests: {total_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())