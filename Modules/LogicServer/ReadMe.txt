this game server is designed for thread rendering and live migration.
the render should aware that the migration only occurs when a frame is started, 
that is to say, there should be the begin/end scene command pair inside a frame.