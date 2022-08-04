package gpt3

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"os"
	"time"
)

// Engine Types
const (
	TextAda001Engine     = "text-ada-001"
	TextBabbage001Engine = "text-babbage-001"
	TextCurie001Engine   = "text-curie-001"
	TextDavinci001Engine = "text-davinci-001"
	AdaEngine            = "ada"
	BabbageEngine        = "babbage"
	CurieEngine          = "curie"
	DavinciEngine        = "davinci"
	DefaultEngine        = DavinciEngine
)

const (
	defaultBaseURL        = "https://api.openai.com/v1"
	defaultUserAgent      = "go-gpt3"
	defaultTimeoutSeconds = 30
)

const (
	FineTunePurpose        = "fine-tune"
	SearchPurpose          = "search"
	AnswersPurpose         = "answers"
	ClassificationsPurpose = "classifications"
)

func getEngineURL(engine string) string {
	return fmt.Sprintf("%s/engines/%s/completions", defaultBaseURL, engine)
}

// A Client is an API client to communicate with the OpenAI gpt-3 APIs
type Client interface {
	// Engines lists the currently available engines, and provides basic information about each
	// option such as the owner and availability.
	Engines(ctx context.Context) (*EnginesResponse, error)

	// Engine retrieves an engine instance, providing basic information about the engine such
	// as the owner and availability.
	Engine(ctx context.Context, engine string) (*EngineObject, error)

	// Completion creates a completion with the default engine. This is the main endpoint of the API
	// which auto-completes based on the given prompt.
	Completion(ctx context.Context, request CompletionRequest) (*CompletionResponse, error)

	// CompletionStream creates a completion with the default engine and streams the results through
	// multiple calls to onData.
	CompletionStream(ctx context.Context, request CompletionRequest, onData func(*CompletionResponse)) error

	// CompletionWithEngine is the same as Completion except allows overriding the default engine on the client
	CompletionWithEngine(ctx context.Context, engine string, request CompletionRequest) (*CompletionResponse, error)

	// CompletionStreamWithEngine is the same as CompletionStream except allows overriding the default engine on the client
	CompletionStreamWithEngine(ctx context.Context, engine string, request CompletionRequest, onData func(*CompletionResponse)) error

	// Given a prompt and an instruction, the model will return an edited version of the prompt.
	Edits(ctx context.Context, request EditsRequest) (*EditsResponse, error)

	// Search performs a semantic search over a list of documents with the default engine.
	Search(ctx context.Context, request SearchRequest) (*SearchResponse, error)

	// SearchWithEngine performs a semantic search over a list of documents with the specified engine.
	SearchWithEngine(ctx context.Context, engine string, request SearchRequest) (*SearchResponse, error)

	//UploadFile Uploads a file that contains document(s) to be used across various endpoints/features.
	UploadFile(ctx context.Context, filename string, purpose string) (*FileUploadResponse, error)

	//DeleteFile deletes a file from the server-side storage identified by id
	DeleteFile(ctx context.Context, fileId string) (*FileDeleteResponse, error)

	//CreateFineTune Creates a job that fine-tunes a specified model from a given dataset.
	CreateFineTune(ctx context.Context, fileId string) (*FineTuneResponse, error)

	//CreateFineTuneWithOptions Creates a job that fine-tunes a specified model from a given dataset.
	CreateFineTuneWithOptions(ctx context.Context, fineTuneOptions FineTuneOptions) (*FineTuneResponse, error)

	//GetFineTune Gets info about the fine-tune job.
	GetFineTune(ctx context.Context, id string) (*FineTuneResponse, error)
}

type client struct {
	baseURL       string
	apiKey        string
	userAgent     string
	httpClient    *http.Client
	defaultEngine string
	idOrg         string
}

// NewClient returns a new OpenAI GPT-3 API client. An apiKey is required to use the client
func NewClient(apiKey string, options ...ClientOption) Client {
	httpClient := &http.Client{
		Timeout: time.Duration(defaultTimeoutSeconds * time.Second),
	}

	c := &client{
		userAgent:     defaultUserAgent,
		apiKey:        apiKey,
		baseURL:       defaultBaseURL,
		httpClient:    httpClient,
		defaultEngine: DefaultEngine,
		idOrg:         "",
	}
	for _, o := range options {
		o(c)
	}
	return c
}

func (c *client) Engines(ctx context.Context) (*EnginesResponse, error) {
	req, err := c.newRequest(ctx, "GET", "/engines", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}

	output := new(EnginesResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

func (c *client) Engine(ctx context.Context, engine string) (*EngineObject, error) {
	req, err := c.newRequest(ctx, "GET", fmt.Sprintf("/engines/%s", engine), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}

	output := new(EngineObject)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

func (c *client) Completion(ctx context.Context, request CompletionRequest) (*CompletionResponse, error) {
	return c.CompletionWithEngine(ctx, c.defaultEngine, request)
}

func (c *client) CompletionWithEngine(ctx context.Context, engine string, request CompletionRequest) (*CompletionResponse, error) {
	request.Stream = false
	req, err := c.newRequest(ctx, "POST", fmt.Sprintf("/engines/%s/completions", engine), request)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}

	output := new(CompletionResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

func (c *client) CompletionStream(ctx context.Context, request CompletionRequest, onData func(*CompletionResponse)) error {
	return c.CompletionStreamWithEngine(ctx, c.defaultEngine, request, onData)
}

var dataPrefix = []byte("data: ")
var doneSequence = []byte("[DONE]")

func (c *client) CompletionStreamWithEngine(
	ctx context.Context,
	engine string,
	request CompletionRequest,
	onData func(*CompletionResponse),
) error {
	request.Stream = true
	req, err := c.newRequest(ctx, "POST", fmt.Sprintf("/engines/%s/completions", engine), request)
	if err != nil {
		return err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return err
	}

	reader := bufio.NewReader(resp.Body)
	defer resp.Body.Close()

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			return err
		}
		// make sure there isn't any extra whitespace before or after
		line = bytes.TrimSpace(line)
		// the completion API only returns data events
		if !bytes.HasPrefix(line, dataPrefix) {
			continue
		}
		line = bytes.TrimPrefix(line, dataPrefix)

		// the stream is completed when terminated by [DONE]
		if bytes.HasPrefix(line, doneSequence) {
			break
		}
		output := new(CompletionResponse)
		if err := json.Unmarshal(line, output); err != nil {
			return fmt.Errorf("invalid json stream data: %v", err)
		}
		onData(output)
	}

	return nil
}

func (c *client) Edits(ctx context.Context, request EditsRequest) (*EditsResponse, error) {
	req, err := c.newRequest(ctx, "POST", "/edits", request)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}

	output := new(EditsResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

func (c *client) Search(ctx context.Context, request SearchRequest) (*SearchResponse, error) {
	return c.SearchWithEngine(ctx, c.defaultEngine, request)
}

func (c *client) SearchWithEngine(ctx context.Context, engine string, request SearchRequest) (*SearchResponse, error) {
	req, err := c.newRequest(ctx, "POST", fmt.Sprintf("/engines/%s/search", engine), request)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}
	output := new(SearchResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

//UploadFile Uploads a file that contains document(s) to be used across various endpoints/features.
func (c *client) UploadFile(ctx context.Context, filename string, purpose string) (*FileUploadResponse, error) {

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", file.Name())
	if err != nil {
		return nil, err
	}

	io.Copy(part, file)

	part, err = writer.CreateFormField("purpose")
	if err != nil {
		return nil, err
	}
	part.Write([]byte(purpose))
	writer.Close()

	url := c.baseURL + "/files"
	req, err := http.NewRequestWithContext(ctx, "POST", url, body)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}
	output := new(FileUploadResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

//DeleteFile deletes a file
func (c *client) DeleteFile(ctx context.Context, fileId string) (*FileDeleteResponse, error) {
	req, err := c.newRequest(ctx, "DELETE", fmt.Sprintf("/files/%s", fileId), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}
	output := new(FileDeleteResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

//CreateFineTune Creates a job that fine-tunes a specified model from a given dataset.
func (c *client) CreateFineTune(ctx context.Context, training_file string) (*FineTuneResponse, error) {
	payload := FineTuneOptions{
		TrainingFile: training_file,
	}
	return c.CreateFineTuneWithOptions(ctx, payload)
}

//CreateFineTuneWithOptions Creates a job that fine-tunes a specified model from a given dataset.
func (c *client) CreateFineTuneWithOptions(ctx context.Context, fineTuneOptions FineTuneOptions) (*FineTuneResponse, error) {

	req, err := c.newRequest(ctx, "POST", "/fine-tunes", fineTuneOptions)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}
	output := new(FineTuneResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

//GetFineTune Gets info about the fine-tune job.
func (c *client) GetFineTune(ctx context.Context, jobId string) (*FineTuneResponse, error) {
	req, err := c.newRequest(ctx, "GET", fmt.Sprintf("/fine-tunes/%s", jobId), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.performRequest(req)
	if err != nil {
		return nil, err
	}
	output := new(FineTuneResponse)
	if err := getResponseObject(resp, output); err != nil {
		return nil, err
	}
	return output, nil
}

func (c *client) performRequest(req *http.Request) (*http.Response, error) {
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if err := checkForSuccess(resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// returns an error if this response includes an error.
func checkForSuccess(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read from body: %w", err)
	}
	var result APIErrorResponse
	if err := json.Unmarshal(data, &result); err != nil {
		// if we can't decode the json error then create an unexpected error
		apiError := APIError{
			StatusCode: resp.StatusCode,
			Type:       "Unexpected",
			Message:    string(data),
		}
		return apiError
	}
	result.Error.StatusCode = resp.StatusCode
	return result.Error
}

func getResponseObject(rsp *http.Response, v interface{}) error {
	defer rsp.Body.Close()
	if err := json.NewDecoder(rsp.Body).Decode(v); err != nil {
		return fmt.Errorf("invalid json response: %w", err)
	}
	return nil
}

func jsonBodyReader(body interface{}) (io.Reader, error) {
	if body == nil {
		return bytes.NewBuffer(nil), nil
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed encoding json: %w", err)
	}
	return bytes.NewBuffer(raw), nil
}

func (c *client) newRequest(ctx context.Context, method, path string, payload interface{}) (*http.Request, error) {
	bodyReader, err := jsonBodyReader(payload)
	if err != nil {
		return nil, err
	}
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, err
	}
	if len(c.idOrg) > 0 {
		req.Header.Set("OpenAI-Organization", c.idOrg)
	}
	req.Header.Set("Content-type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	return req, nil
}
