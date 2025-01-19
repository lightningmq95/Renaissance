# Renaissance

## Server Setup

1. Navigate to the `server` folder:
    ```sh
    cd server
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the FastAPI server:
    ```sh
    fastapi dev
    ```

## Client Setup

1. In a new terminal, Navigate to the `client` folder:
    ```sh
    cd client
    ```
2. Download and install `npm` on your machine if you haven't already from:
    https://nodejs.org/en/download/

3. Install `pnpm` on your machine if you haven't already:
    ```sh
    npm install -g pnpm@latest-10
    ```

4. Install the dependencies:
    ```sh
    pnpm install
    ```

5. Run the development server:
    ```sh
    pnpm run dev
    ```

Once both the server and client are up and running, you should be able to access the application in your browser.